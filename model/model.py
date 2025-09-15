import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, LayerNorm as PygLayerNorm
from torch_geometric.utils import add_self_loops

# -------------------------
class GINEBlock(nn.Module):
    def __init__(self, hidden_dim, edge_emb_dim=None, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv = GINEConv(self.mlp, edge_dim=hidden_dim)
        self.ln = PygLayerNorm(hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        h_in = x
        h = self.conv(x, edge_index, edge_attr)
        if h.shape == h_in.shape:
            h = h + h_in
        h = self.ln(h)
        h = self.act(h)
        h = self.dropout(h)
        return torch.nan_to_num(h)

# -------------------------
class SubgraphMultiheadPool(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, tau=0.8, topk=4):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.h = num_heads
        self.d = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.tau = nn.Parameter(torch.tensor(float(tau)), requires_grad=False)
        self.topk = topk
        self.sg_query = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, x, subgraph_masks, center_idx=None):
        S, N = subgraph_masks.size(0), x.size(0)
        H = x.size(1)
        if center_idx is not None:
            q0 = x[center_idx]
        else:
            q0 = x.mean(dim=0, keepdim=True).expand(S, -1)

        Q = self.q_proj(q0).view(S, self.h, self.d)
        K = self.k_proj(x).view(N, self.h, self.d)
        V = self.v_proj(x).view(N, self.h, self.d)

        scores = torch.einsum('shd,nhd->snh', Q, K) / (self.d ** 0.5 * self.tau)
        mask = subgraph_masks.unsqueeze(-1).expand(-1, -1, self.h)
        scores = scores.masked_fill(~mask, -1e9)

        attn = F.softmax(scores, dim=1)
        attn = torch.nan_to_num(attn, nan=0.0)
        node_attn = attn.mean(dim=-1)

        sub = torch.einsum('snh,nhd->shd', attn, V).contiguous().view(S, H)
        sub = self.o_proj(sub)

        sg_scores = (sub @ self.sg_query) / (H ** 0.5)
        return torch.nan_to_num(sub), node_attn, torch.nan_to_num(sg_scores)

    def topk_aggregate(self, sub, scores, k):
        if sub.size(0) == 0:
            h = sub.new_zeros((sub.shape[-1],))
            return h, (torch.empty(0, dtype=torch.long), sub.new_zeros((0,)))
        k = min(k if k is not None else self.topk, sub.size(0))
        vals, idx = torch.topk(scores, k)
        w = F.softmax(vals, dim=0)
        return (sub[idx] * w.unsqueeze(-1)).sum(0), (idx, w)

# -------------------------
class Main_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        H = self.hidden_dim = cfg.SUB.DIM
        self.num_atom_types = cfg.MODEL.NUM_ATOM_TYPES
        self.num_fingerprint_types = cfg.MODEL.NUM_FINGERPRINT_TYPES
        self.num_physicochemical = cfg.MODEL.NUM_PHYSICOCHEMICAL
        P = self.num_physicochemical
        self.num_gnn_layers = cfg.SUB.LAYER_GNN
        self.dropout_rate = cfg.MODEL.DROPOUT

        # ==== Embeddings ====
        self.atom_embedding = nn.Embedding(self.num_atom_types, H)
        self.fingerprint_embedding = nn.Embedding(self.num_fingerprint_types, H)
        self.num_bond_types = getattr(cfg.MODEL, "NUM_BOND_TYPES", 8)
        self.bond_embedding = nn.Embedding(self.num_bond_types, H)

        # 输入投影
        self.proj_in = nn.Linear(2 * H, H)   # 溶剂：需要 atom_emb 与 fingerprint_emb 拼接
        self.salt_proj_in = nn.Linear(H, H)  # 盐：只有 atom_emb，没有 fingerprint

        # ==== GNN encoder (共享于溶剂与盐) ====
        self.gnn_layers = nn.ModuleList([GINEBlock(H, dropout=self.dropout_rate)
                                         for _ in range(self.num_gnn_layers)])
        self.dropout = nn.Dropout(self.dropout_rate)
        self.act = nn.SiLU()

        # ==== Subgraph pooling（只用于溶剂） ====
        self.subgraph_pool = SubgraphMultiheadPool(H, num_heads=4, tau=0.8, topk=4)

        # ==== Mixture-level bilinear attention（原逻辑） ====
        self.key_mlp = nn.Sequential(
            nn.Linear(P + 1, H),
            nn.SiLU(),
            nn.Linear(H, H),
        )
        self.bilinear = nn.Bilinear(H, H, 1, bias=False)
        self.score_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(H)), requires_grad=True)

        self.use_w_prior   = getattr(getattr(cfg, "SIMPLE", object()), "USE_W_PRIOR", True)
        self.w_prior_beta  = nn.Parameter(torch.tensor(
            float(getattr(getattr(cfg, "SIMPLE", object()), "W_PRIOR_BETA", 0.2))
        ), requires_grad=False)

        # Fusion & gate（
        self.simple_proj = nn.Linear(H, H, bias=False)
        gate_init = float(getattr(getattr(cfg, "SIMPLE", object()), "GATE_INIT", -0.5))
        self.simple_mix_gate = nn.Parameter(torch.tensor(gate_init))

        # ==== Salt  ====
        self.salt_graph_proj = nn.Linear(H, H)
        self.salt_feat_proj  = nn.Linear(2, H)      # [salt_conc, temperature]
        self.salt_gate       = nn.Parameter(torch.tensor(0.0))  

        # ==== Head ====
        self.head = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, H),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(H, 1),
        )

        self.apply(self.init_weights)

    # adjacency -> edge_index (+ self loops) & edge_attr ids
    def _edges_from_adjacency(self, adjacency, num_nodes, device):
        nz = adjacency.nonzero(as_tuple=False)
        if nz.numel() == 0 and num_nodes > 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr_ids = torch.empty((0,), dtype=torch.long, device=device)
        else:
            edge_index = nz.t().contiguous()
            valid = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            edge_index = edge_index[:, valid]
            edge_attr_ids = adjacency[edge_index[0], edge_index[1]].long()
        E0 = edge_index.size(1)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        E1 = edge_index.size(1)
        num_added = E1 - E0
        if num_added > 0:
            loop_ids = torch.zeros(num_added, dtype=torch.long, device=device)
            edge_attr_ids = torch.cat([edge_attr_ids, loop_ids], dim=0)
        if hasattr(self, "num_bond_types"):
            edge_attr_ids = edge_attr_ids.clamp_min(0).clamp_max(self.num_bond_types - 1)
        return edge_index, edge_attr_ids

    def _encode_salt_graph(self, salt_atom_features, salt_adjacency):
        if salt_atom_features is None or salt_atom_features.numel() == 0:
            H = self.hidden_dim
            return torch.zeros(H, device=self.salt_gate.device)

        device0 = salt_atom_features.device
        af = salt_atom_features.long()
        adj = salt_adjacency.float()

        x = self.atom_embedding(af)                   # [N,H]
        x = torch.nan_to_num(self.salt_proj_in(x))    # [N,H]

        N = x.size(0)
        edge_index, edge_attr_ids = self._edges_from_adjacency(adj, N, device0)
        edge_attr = self.bond_embedding(edge_attr_ids)  # [E,H]

        h = x
        for block in self.gnn_layers:
            h = block(h, edge_index, edge_attr)

        salt_embed = h.mean(dim=0)  # Global mean pooling
        return torch.nan_to_num(salt_embed)

    def forward(self, inputs, ratios):
        batch_size = len(inputs)
        outputs, attn_all = [], []

        for b in range(batch_size):
            solvent_inputs = inputs[b]
            atom_features_list   = solvent_inputs['atom_features']
            adjacency_list       = solvent_inputs['adjacency']
            physico_list         = solvent_inputs['physicochemical']
            fingerprint_list     = solvent_inputs['fingerprints']
            subgraph_masks_list  = solvent_inputs['subgraph_masks']

            device0 = atom_features_list[0].device if isinstance(atom_features_list[0], torch.Tensor) \
                      else torch.device('cpu')
            w = torch.as_tensor(ratios[b], dtype=torch.float32, device=device0)  # [M]

            mol_embeds, mol_dbg, pv_list = [], [], []

            # ----- per-solvent encoding -----
            for (atom_features, adjacency, physico, fingerprint, sub_masks) in zip(
                atom_features_list, adjacency_list, physico_list, fingerprint_list, subgraph_masks_list
            ):
                af  = atom_features if isinstance(atom_features, torch.Tensor) else torch.as_tensor(atom_features, dtype=torch.long,   device=device0)
                adj = adjacency     if isinstance(adjacency,     torch.Tensor) else torch.as_tensor(adjacency,     dtype=torch.float32, device=device0)
                fp  = fingerprint   if isinstance(fingerprint,   torch.Tensor) else torch.as_tensor(fingerprint,   dtype=torch.long,   device=device0)
                pv  = physico       if isinstance(physico,       torch.Tensor) else torch.as_tensor(physico,       dtype=torch.float32, device=device0)
                sgm = sub_masks     if isinstance(sub_masks,     torch.Tensor) else torch.as_tensor(sub_masks,     dtype=torch.bool,    device=device0)

                pv = torch.nan_to_num(pv, nan=0.0, posinf=1e3, neginf=-1e3)
                pv_list.append(pv)

                x = torch.cat([self.atom_embedding(af), self.fingerprint_embedding(fp)], dim=-1)  # [N,2H]
                x = torch.nan_to_num(self.proj_in(x))

                N = x.size(0)
                edge_index, edge_attr_ids = self._edges_from_adjacency(adj, N, device0)
                edge_attr = self.bond_embedding(edge_attr_ids)

                h = x
                for block in self.gnn_layers:
                    h = block(h, edge_index, edge_attr)

                sub_embeds, node_attn, sg_scores = self.subgraph_pool(h, sgm, center_idx=None)
                mol_embed, (idx_top, w_top) = self.subgraph_pool.topk_aggregate(sub_embeds, sg_scores, k=None)

                sub_embeds, node_attn, sg_scores = self.subgraph_pool(h, sgm, center_idx=None)
                mol_embed, (idx_top, w_top) = self.subgraph_pool.topk_aggregate(sub_embeds, sg_scores, k=None)

                beta_full = F.softmax(sg_scores, dim=0)
                beta_topk = torch.zeros_like(beta_full)
                beta_topk[idx_top] = F.softmax(sg_scores[idx_top], 0)

                mol_embeds.append(mol_embed)
                mol_dbg.append({
                    'subgraph_masks': sgm.detach().cpu(),
                    'subgraph_node_attn': node_attn.detach().cpu(),
                    'subgraph_scores': sg_scores.detach().cpu(),
                    'beta_full': beta_full.detach().cpu(),
                    'beta_topk': beta_topk.detach().cpu(),
                    'subgraph_topk_idx': idx_top.detach().cpu(),
                    'subgraph_topk_w': w_top.detach().cpu(),
                })

            # ----- mixture-level attention-----
            M = len(mol_embeds)
            mol_embeds  = torch.stack(mol_embeds)              # [M, H]
            prop_tensor = torch.stack(pv_list).float()         # [M, P]
            prop_tensor = torch.nan_to_num(prop_tensor, nan=0.0, posinf=1e3, neginf=-1e3)

            key_in = torch.cat([prop_tensor, w.view(-1, 1)], dim=-1)  # [M, P+1]
            key_m  = self.key_mlp(key_in)                             # [M, H]

            scores = self.bilinear(mol_embeds, key_m).squeeze(-1) * self.score_scale
            eps = 1e-8
            if self.use_w_prior:
                w_norm = w / (w.sum() + eps)
                scores = scores + self.w_prior_beta * torch.log(torch.clamp(w_norm, min=eps))

            alpha = F.softmax(scores, dim=0)
            alpha = torch.nan_to_num(alpha, nan=0.0)

            weighted_embedding = torch.sum(alpha.unsqueeze(-1) * mol_embeds, dim=0)  # [H]
            w_lin = w / (w.sum() + eps)
            simple_mix = torch.sum(w_lin.unsqueeze(-1) * mol_embeds, dim=0)          # [H]
            simple_mix = self.simple_proj(simple_mix)
            gamma = torch.sigmoid(self.simple_mix_gate)
            mixture_embed = (1.0 - gamma) * weighted_embedding + gamma * simple_mix  # [H]

            # ====== Mixture level ======
            salt_af = solvent_inputs.get('salt_atom_features', None)
            salt_adj = solvent_inputs.get('salt_adjacency', None)
            salt_feat = solvent_inputs.get('salt_features', None)   # [2] = [salt_conc, temperature]

            salt_add = 0.0
            if salt_af is not None and salt_adj is not None:
                salt_embed = self._encode_salt_graph(salt_af, salt_adj)            # [H]
                salt_add += self.salt_graph_proj(salt_embed)                       # [H]
            if salt_feat is not None:
                salt_feat_h = self.salt_feat_proj(salt_feat.float().view(-1))      # [H]
                salt_add += salt_feat_h

            if isinstance(salt_add, torch.Tensor):
                mixture_embed = mixture_embed + torch.sigmoid(self.salt_gate) * salt_add  # [H]

            # ----- head -----
            y_hat = self.head(mixture_embed).squeeze()
            outputs.append(y_hat)

            alpha_entropy = -(alpha * torch.clamp(alpha, min=1e-8).log()).sum()
            attn_all.append({
                'mixture_attention': alpha.detach().cpu(),
                'mixture_scores': scores.detach().cpu(),
                'gamma': gamma.detach().cpu() if 'gamma' in locals() else None,
                'molecules': mol_dbg,
            })

        y = torch.stack(outputs).view(batch_size, 1)
        y = torch.nan_to_num(y)
        return y, attn_all

    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, PygLayerNorm)):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
