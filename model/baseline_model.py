import torch
import torch.nn as nn
import torch.nn.functional as F

def _to_tensor(x, dtype, device):
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=dtype, device=device)

def _normalize_weights(w):
    eps = 1e-8
    return w / (w.sum() + eps)

def _stack_safe(tensors, dim=0):
    if len(tensors) == 1:
        return tensors[0].unsqueeze(dim)
    return torch.stack(tensors, dim=dim)

# ------------- Baseline 1: SimpleMeanMix -------------
class SimpleMeanMix(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        H = self.hidden_dim = cfg.SUB.DIM
        self.num_atom_types = cfg.MODEL.NUM_ATOM_TYPES
        self.num_fp_types   = cfg.MODEL.NUM_FINGERPRINT_TYPES
        self.num_phys       = cfg.MODEL.NUM_PHYSICOCHEMICAL
        P = self.num_phys

        self.atom_embedding = nn.Embedding(self.num_atom_types, H)
        self.fp_embedding   = nn.Embedding(self.num_fp_types, H)
        self.proj_in = nn.Linear(2*H, H)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(cfg.MODEL.DROPOUT)

        self.head = nn.Sequential(
            nn.LayerNorm(H + P),
            nn.Linear(H + P, H),
            nn.ReLU(),
            nn.Dropout(cfg.MODEL.DROPOUT),
            nn.Linear(H, 1)
        )
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def _mol_embed(self, af, fp):
        x = torch.cat([self.atom_embedding(af), self.fp_embedding(fp)], dim=-1)  # [N,2H]
        x = self.act(self.proj_in(x))                                            # [N,H]
        return x.mean(dim=0)                                                     # [H]

    def forward(self, inputs, ratios):
        B = len(inputs)
        outs, infos = [], []
        for b in range(B):
            pack = inputs[b]
            af_list  = pack['atom_features']
            fp_list  = pack['fingerprints']
            phys_list= pack['physicochemical']

            device0 = af_list[0].device if isinstance(af_list[0], torch.Tensor) else torch.device('cpu')
            mol_vecs, phys = [], []
            for af, fp, pv in zip(af_list, fp_list, phys_list):
                af = _to_tensor(af, torch.long,  device0)
                fp = _to_tensor(fp, torch.long,  device0)
                pv = _to_tensor(pv, torch.float, device0)
                pv = torch.nan_to_num(pv, nan=0.0, posinf=1e3, neginf=-1e3)
                mol_vecs.append(self._mol_embed(af, fp))  # [H]
                phys.append(pv)                            # [P]

            M = len(mol_vecs)
            mol_mat  = _stack_safe(mol_vecs, dim=0)        # [M,H]
            phys_mat = _stack_safe(phys,    dim=0)         # [M,P]

            w = _to_tensor(ratios[b], torch.float, device0).view(-1)
            if w.numel() != M:
                m = min(M, w.numel())
                mol_mat  = mol_mat[:m]
                phys_mat = phys_mat[:m]
                w = w[:m]; M = m
            w = _normalize_weights(w)                      # [M]

            mix_h = (w.unsqueeze(-1) * mol_mat).sum(0)     # [H]
            mix_p = (w.unsqueeze(-1) * phys_mat).sum(0)    # [P]

            z = torch.cat([mix_h, mix_p], dim=-1)          # [H+P]
            y = self.head(z).squeeze()                     # []
            outs.append(y)
            infos.append({'w': w.detach().cpu()})

        y_all = _stack_safe(outs, dim=0).view(B, 1)
        return torch.nan_to_num(y_all), infos

# ------------- Baseline 2: GCNMeanMix -------------
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops

class GCNMeanMix(nn.Module):
    def __init__(self, cfg, num_layers=2):
        super().__init__()
        H = self.hidden_dim = cfg.SUB.DIM
        self.num_atom_types = cfg.MODEL.NUM_ATOM_TYPES
        self.num_fp_types   = cfg.MODEL.NUM_FINGERPRINT_TYPES
        self.num_phys       = cfg.MODEL.NUM_PHYSICOCHEMICAL
        P = self.num_phys

        self.atom_embedding = nn.Embedding(self.num_atom_types, H)
        self.fp_embedding   = nn.Embedding(self.num_fp_types, H)
        self.proj_in = nn.Linear(2*H, H)

        self.gcn = nn.ModuleList([GCNConv(H, H) for _ in range(num_layers)])
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(cfg.MODEL.DROPOUT)

        self.head = nn.Sequential(
            nn.LayerNorm(H + P),
            nn.Linear(H + P, H),
            nn.ReLU(),
            nn.Dropout(cfg.MODEL.DROPOUT),
            nn.Linear(H, 1)
        )
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def _edges_from_adj(self, adj, N, device):
        nz = adj.nonzero(as_tuple=False)
        edge_index = nz.t().contiguous() if nz.numel() > 0 else torch.empty((2,0), dtype=torch.long, device=device)
        edge_index, _ = add_self_loops(edge_index, num_nodes=N)
        return edge_index

    def _mol_embed(self, af, fp, adj):
        device0 = af.device
        x = torch.cat([self.atom_embedding(af), self.fp_embedding(fp)], dim=-1)  # [N,2H]
        x = self.act(self.proj_in(x))                                            # [N,H]
        N = x.size(0)
        edge_index = self._edges_from_adj(adj, N, device0)
        for conv in self.gcn:
            x = self.act(conv(x, edge_index))
        return x.mean(dim=0)                                                     # [H]

    def forward(self, inputs, ratios):
        B = len(inputs)
        outs, infos = [], []
        for b in range(B):
            pack = inputs[b]
            af_list  = pack['atom_features']
            fp_list  = pack['fingerprints']
            adj_list = pack['adjacency']
            phys_list= pack['physicochemical']

            device0 = _to_tensor(af_list[0], torch.long, 'cpu').device if not isinstance(af_list[0], torch.Tensor) else af_list[0].device

            mol_vecs, phys = [], []
            for af, fp, adj, pv in zip(af_list, fp_list, adj_list, phys_list):
                af  = _to_tensor(af,  torch.long,  device0)
                fp  = _to_tensor(fp,  torch.long,  device0)
                adj = _to_tensor(adj, torch.float, device0)
                pv  = _to_tensor(pv,  torch.float, device0)
                pv  = torch.nan_to_num(pv, nan=0.0, posinf=1e3, neginf=-1e3)
                mol_vecs.append(self._mol_embed(af, fp, adj))  # [H]
                phys.append(pv)                                # [P]

            M = len(mol_vecs)
            mol_mat  = _stack_safe(mol_vecs, dim=0)  # [M,H]
            phys_mat = _stack_safe(phys,    dim=0)   # [M,P]

            w = _to_tensor(ratios[b], torch.float, device0).view(-1)
            if w.numel() != M:
                m = min(M, w.numel())
                mol_mat  = mol_mat[:m]
                phys_mat = phys_mat[:m]
                w = w[:m]; M = m
            w = _normalize_weights(w)

            mix_h = (w.unsqueeze(-1) * mol_mat).sum(0)   # [H]
            mix_p = (w.unsqueeze(-1) * phys_mat).sum(0)  # [P]
            z = torch.cat([mix_h, mix_p], dim=-1)
            y = self.head(z).squeeze()
            outs.append(y)
            infos.append({'w': w.detach().cpu()})

        y_all = _stack_safe(outs, dim=0).view(B, 1)
        return torch.nan_to_num(y_all), infos

# ------------- Baseline 3: DeepSetsMix -------------
class DeepSetsMix(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        H = self.hidden_dim = cfg.SUB.DIM
        self.num_atom_types = cfg.MODEL.NUM_ATOM_TYPES
        self.num_fp_types   = cfg.MODEL.NUM_FINGERPRINT_TYPES
        self.num_phys       = cfg.MODEL.NUM_PHYSICOCHEMICAL
        P = self.num_phys

        self.atom_embedding = nn.Embedding(self.num_atom_types, H)
        self.fp_embedding   = nn.Embedding(self.num_fp_types, H)
        self.proj_in = nn.Linear(2*H, H)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(cfg.MODEL.DROPOUT)

        self.phi = nn.Sequential(
            nn.LayerNorm(H + P + 1),
            nn.Linear(H + P + 1, H),
            nn.ReLU(),
            nn.Linear(H, H)
        )
        # rho:
        self.rho = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Dropout(cfg.MODEL.DROPOUT),
            nn.Linear(H, 1)
        )
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def _mol_embed(self, af, fp):
        x = torch.cat([self.atom_embedding(af), self.fp_embedding(fp)], dim=-1)  # [N,2H]
        return self.act(self.proj_in(x)).mean(dim=0)                              # [H]

    def forward(self, inputs, ratios):
        B = len(inputs)
        outs, infos = [], []
        for b in range(B):
            pack = inputs[b]
            af_list  = pack['atom_features']
            fp_list  = pack['fingerprints']
            phys_list= pack['physicochemical']

            device0 = af_list[0].device if isinstance(af_list[0], torch.Tensor) else torch.device('cpu')

            mol_vecs, phys = [], []
            for af, fp, pv in zip(af_list, fp_list, phys_list):
                af = _to_tensor(af, torch.long,  device0)
                fp = _to_tensor(fp, torch.long,  device0)
                pv = _to_tensor(pv, torch.float, device0)
                pv = torch.nan_to_num(pv, nan=0.0, posinf=1e3, neginf=-1e3)
                mol_vecs.append(self._mol_embed(af, fp))  # [H]
                phys.append(pv)                            # [P]

            M = len(mol_vecs)
            mol_mat  = _stack_safe(mol_vecs, dim=0)        # [M,H]
            phys_mat = _stack_safe(phys,    dim=0)         # [M,P]
            w = _to_tensor(ratios[b], torch.float, device0).view(-1)
            if w.numel() != M:
                m = min(M, w.numel())
                mol_mat  = mol_mat[:m]
                phys_mat = phys_mat[:m]
                w = w[:m]; M = m
            w = _normalize_weights(w)

            # Deep Sets：sum_i phi([h_i, p_i, w_i])
            phi_in = torch.cat([mol_mat, phys_mat, w.view(-1,1)], dim=-1)  # [M, H+P+1]
            summed = self.phi(phi_in).sum(dim=0)                            # [H]
            y = self.rho(summed).squeeze()
            outs.append(y)
            infos.append({'w': w.detach().cpu()})

        y_all = _stack_safe(outs, dim=0).view(B, 1)
        return torch.nan_to_num(y_all), infos

# ------------- Baseline 4: PhysChemLinearMix -------------
class PhysChemLinearMix(nn.Module):

    def __init__(self, cfg, hidden=64):
        super().__init__()
        P = cfg.MODEL.NUM_PHYSICOCHEMICAL
        self.head = nn.Sequential(
            nn.LayerNorm(P),
            nn.Linear(P, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, inputs, ratios):
        B = len(inputs)
        outs, infos = [], []
        for b in range(B):
            phys_list = inputs[b]['physicochemical']
            device0 = phys_list[0].device if isinstance(phys_list[0], torch.Tensor) else torch.device('cpu')
            phys = [_to_tensor(p, torch.float, device0) for p in phys_list]  # M*[P]
            phys_mat = _stack_safe(phys, dim=0)                              # [M,P]

            w = _to_tensor(ratios[b], torch.float, device0).view(-1)
            if w.numel() != phys_mat.size(0):
                m = min(phys_mat.size(0), w.numel())
                phys_mat = phys_mat[:m]; w = w[:m]
            w = _normalize_weights(w)

            mix_p = (w.unsqueeze(-1) * phys_mat).sum(0)  # [P]
            y = self.head(mix_p).squeeze()
            outs.append(y)
            infos.append({'w': w.detach().cpu()})
        y_all = _stack_safe(outs, dim=0).view(B, 1)
        return torch.nan_to_num(y_all), infos
