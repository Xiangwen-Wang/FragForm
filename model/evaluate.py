import os
import glob
import timeit
import pickle
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import mean_squared_error, mean_absolute_error

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from configs import get_cfg_defaults
from model import Main_model

cfg = get_cfg_defaults()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# PKL cache + vocabs
# -----------------------------
_PKL = None
def _load_pkl(preprocessed_path):
    global _PKL
    if _PKL is None:
        with open(preprocessed_path, 'rb') as f:
            _PKL = pickle.load(f)
        print(f"[PKL] loaded: {len(_PKL['results'])} molecules; "
              f"atom_dict={len(_PKL.get('atom_dict', {}))}, bond_dict={len(_PKL.get('bond_dict', {}))}")
    return _PKL

_MOLECULE_DATA = None
def load_preprocessed_data(preprocessed_path):
    global _MOLECULE_DATA
    if _MOLECULE_DATA is None:
        _MOLECULE_DATA = _load_pkl(preprocessed_path)['results']
        print(f"[MOLCACHE] {len(_MOLECULE_DATA)} molecules in cache")
    return _MOLECULE_DATA

def _get_vocabs(preprocessed_path):
    pkl = _load_pkl(preprocessed_path)
    return pkl.get('atom_dict', {}), pkl.get('bond_dict', {})

# -----------------------------
# Utils
# -----------------------------
def _find_col(df, candidates):
    cl = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cl:
            return cl[cand.lower()]
    return None

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

# -----------------------------
# On-the-fly salt graph 
# -----------------------------
def _encode_atom_features_from_vocab(mol, atom_dict, num_atom_types):
    syms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    for atom in mol.GetAromaticAtoms():
        idx = atom.GetIdx()
        syms[idx] = (syms[idx], 'aromatic')
    feats = []
    for s in syms:
        idx = atom_dict.get(s, 0)
        if idx >= num_atom_types:
            idx = 0
        feats.append(idx)
    return np.array(feats, dtype=np.int64)

def _encode_bond_adj_from_vocab(mol, bond_dict):
    n = mol.GetNumAtoms()
    adj = np.zeros((n, n), dtype=np.float32)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bt = bond_dict.get(str(b.GetBondType()), 0)
        adj[i, j] = adj[j, i] = float(bt)
    return adj

def build_salt_graph_from_smiles(salt_smiles, preprocessed_path):
    if not salt_smiles or str(salt_smiles).lower() in ('nan', 'none'):
        return torch.zeros((0,), dtype=torch.long, device=device), torch.zeros((0,0), dtype=torch.float, device=device)
    mol = Chem.MolFromSmiles(salt_smiles)
    if mol is None:
        return torch.zeros((0,), dtype=torch.long, device=device), torch.zeros((0,0), dtype=torch.float, device=device)
    mol = Chem.AddHs(mol)
    atom_dict, bond_dict = _get_vocabs(preprocessed_path)
    af = _encode_atom_features_from_vocab(mol, atom_dict, cfg.MODEL.NUM_ATOM_TYPES)
    adj = _encode_bond_adj_from_vocab(mol, bond_dict)
    return (torch.tensor(af, dtype=torch.long, device=device),
            torch.tensor(adj, dtype=torch.float, device=device))

# -----------------------------
# Physchem proxy 
# -----------------------------
def estimate_physchem_proxy(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return dict(Tm=0.0, Tb=100.0, FP=20.0, Visc=1.0, EPS=2.0)
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    Tm = 0.15 * mw + 0.30 * rings + 6.0 * hbd + 3.0 * hba - 1.8 * rot - 10.0
    Tb = 0.45 * mw + 0.06 * tpsa + 4.0 * rings + 3.0 * hbd - 60.0
    expr = -3.5 + 0.012 * mw + 0.008 * tpsa + 0.25 * hbd
    expr = float(np.clip(expr, -10.0, 10.0))
    Visc = float(np.exp(expr))
    FP = 0.42 * Tb - 35.0
    EPS = 1.0 + 0.035 * tpsa + 0.45 * hbd + 0.35 * hba - 0.12 * logp
    Tm = float(np.clip(Tm, -273.15, 2000.0))
    Tb = float(np.clip(Tb, -273.15, 3000.0))
    FP = float(np.clip(FP, -200.0, 1200.0))
    Visc = float(np.clip(Visc, 1e-5, 1e5))
    EPS = float(np.clip(EPS, 1.0, 200.0))
    return dict(Tm=Tm, Tb=Tb, FP=FP, Visc=Visc, EPS=EPS)

# -----------------------------
# Build inputs
# -----------------------------
def prepare_inputs2(solvent_names, ratios, physico_values, salt_smiles, salt_conc, temperature):
    molecule_data = load_preprocessed_data(cfg.DIR.PREPROCESSED_DIR)

    atom_features_list = []
    adjacency_list = []
    fingerprint_list = []
    physico_list = []
    subgraph_masks_list = []

    for idx, smiles in enumerate(solvent_names):
        smi = smiles.strip()
        if smi not in molecule_data:
            return None
        entry = molecule_data[smi]
        if entry.get('fingerprints', None) is None or entry.get('subgraph_masks', None) is None:
            return None

        af  = torch.tensor(entry['atom_features'], dtype=torch.long,  device=device)
        adj = torch.tensor(entry['adjacency'],     dtype=torch.float, device=device)
        fp  = torch.tensor(entry['fingerprints'],  dtype=torch.long,  device=device)
        sgm = torch.tensor(entry['subgraph_masks'], dtype=torch.bool, device=device)

        pv = physico_values[idx]  
        mm_t = torch.tensor([float(x) for x in pv], dtype=torch.float, device=device)
        mm_t = torch.nan_to_num(mm_t, nan=0.0, posinf=1e3, neginf=-1e3)

        atom_features_list.append(af)
        adjacency_list.append(adj)
        fingerprint_list.append(fp)
        physico_list.append(mm_t)
        subgraph_masks_list.append(sgm)


    salt_af_t, salt_adj_t = build_salt_graph_from_smiles(salt_smiles, cfg.DIR.PREPROCESSED_DIR)
    salt_feat = torch.tensor([float(salt_conc), float(temperature)], dtype=torch.float, device=device)
    salt_feat = torch.nan_to_num(salt_feat, nan=0.0, posinf=1e3, neginf=-1e3)

    return {
        'atom_features': atom_features_list,
        'adjacency': adjacency_list,
        'fingerprints': fingerprint_list,
        'physicochemical': physico_list,
        'subgraph_masks': subgraph_masks_list,
        'salt_atom_features': salt_af_t,
        'salt_adjacency':     salt_adj_t,
        'salt_features':      salt_feat,
    }

# -----------------------------
# Dataset loader
# -----------------------------
def load_dataset2(csv_path, preprocessed_path):
    df = pd.read_csv(csv_path)
    if df.shape[1] == 1:
        df = pd.read_csv(csv_path, sep="\t")

    pairs = []
    for i in range(1, 5 + 1):
        s_col, c_col = f"substrate{i}", f"con{i}"
        if s_col in df.columns and c_col in df.columns:
            pairs.append((s_col, c_col))
    if not pairs:
        raise KeyError("No andy substrateX / conX column（X=1..5）")
    if "ionic" not in df.columns:
        raise KeyError("no column named：ionic")

    # 额外列
    salt_col = _find_col(df, ["salt", "Salt", "SALT"])
    temp_col = _find_col(df, ["temperature", "Temperature", "temp", "Temp"])
    salt_conc_col = _find_col(df, [
        "salt_con", "salt_conc", "salt concentration", "salt_concentration",
        "SaltConc", "Salt_Conc", "c_salt", "salt_molarity", "salt_molality"
    ])
    if salt_col is None:
        raise KeyError("no column named：Salt（salt/SALT）")
    if temp_col is None:
        raise KeyError("no column named：Temperature（temperature/temp）")

    data, skipped = [], 0
    for _, row in df.iterrows():
        comps = []  # [(smiles, con, mw)]
        for s_col, c_col in pairs:
            s_raw = row[s_col]
            c_raw = row[c_col]
            s = "" if pd.isna(s_raw) else str(s_raw).strip()
            try:
                c = float(c_raw)
            except Exception:
                c = np.nan

            if not s or s.lower() in ("nan", "none", "o"):
                continue
            if not np.isfinite(c) or c <= 0:
                continue

            mol = Chem.MolFromSmiles(s)
            if mol is None:
                comps = []; break
            mw = Descriptors.MolWt(mol)
            if not np.isfinite(mw) or mw <= 0:
                comps = []; break
            comps.append((s, c, mw))

        if not comps:
            skipped += 1; continue

        cons = np.array([c for (_, c, _) in comps], dtype=float)
        if not np.all(np.isfinite(cons)) or cons.sum() <= 0:
            skipped += 1; continue
        ratios = (cons / cons.sum()).tolist()
        names  = [s for (s, _, _) in comps]

        phys_list, ok = [], True
        for (s, _, mw) in comps:
            prop = estimate_physchem_proxy(s)
            pvec = [prop['Tm'], prop['Tb'], prop['FP'], prop['Visc'], prop['EPS'], float(mw)]
            if not all(np.isfinite(pvec)) or len(pvec) != cfg.MODEL.NUM_PHYSICOCHEMICAL:
                ok = False; break
            phys_list.append(pvec)
        if not ok:
            skipped += 1; continue

        try:
            y = float(row["ionic"])
        except Exception:
            y = np.nan
        if not np.isfinite(y):
            skipped += 1; continue

        salt_smi = str(row[salt_col]).strip() if pd.notna(row[salt_col]) else ""
        temperature = _safe_float(row[temp_col], default=0.0) if pd.notna(row[temp_col]) else 0.0
        salt_conc = 1.0
        if salt_conc_col and pd.notna(row.get(salt_conc_col)):
            salt_conc = _safe_float(row[salt_conc_col], default=1.0)

        data.append((names, ratios, phys_list, y, salt_smi, float(salt_conc), float(temperature)))

    print(f"[DATA] Loaded {len(data)} samples, skipped {skipped}")
    return data

# -----------------------------
# Collate
# -----------------------------
def custom_collate_fn2(batch):
    solvents, ratios, physico, labels = [], [], [], []
    salt_smiles, salt_concs, temperatures = [], [], []
    for s, r, p, y, salt_smi, salt_conc, temperature in batch:
        solvents.append(s); ratios.append(r); physico.append(p); labels.append(y)
        salt_smiles.append(salt_smi); salt_concs.append(salt_conc); temperatures.append(temperature)
    return (solvents, ratios, physico, salt_smiles, salt_concs, temperatures,
            torch.tensor(labels, dtype=torch.float, device=device))

# -----------------------------
# Inference helpers
# -----------------------------
def find_model_path(odir):
    priors = ["best_model2.pt", "best_model_combined.pt", "best_model.pt"]
    for name in priors:
        p = os.path.join(odir, name)
        if os.path.exists(p):
            return p
    pts = sorted(glob.glob(os.path.join(odir, "*.pt")), key=os.path.getmtime, reverse=True)
    if pts:
        return pts[0]
    raise FileNotFoundError("cannot find pt file")

def predict_all(model, loader):
    model.eval()
    trues, preds = [], []
    with torch.no_grad():
        for solvents, ratios, physico, salt_smiles, salt_concs, temperatures, labels in loader:
            batch_inputs, valid_labels, valid_ratios = [], [], []
            for i, (s, r, p, salt_smi, c_salt, temp) in enumerate(zip(
                solvents, ratios, physico, salt_smiles, salt_concs, temperatures
            )):
                li = float(labels[i].item())
                if not np.isfinite(li):
                    continue
                inp = prepare_inputs2(s, r, p, salt_smi, c_salt, temp)
                if inp is None:
                    continue
                batch_inputs.append(inp)
                valid_ratios.append(torch.tensor(r, dtype=torch.float, device=device))
                valid_labels.append(li)
            if not batch_inputs:
                continue
            outputs, _ = model(batch_inputs, valid_ratios)
            preds.extend(outputs.view(-1).detach().cpu().numpy().tolist())
            trues.extend(valid_labels)
    return np.asarray(trues, dtype=float), np.asarray(preds, dtype=float)

def compute_metrics_log10(y, yhat):
    ok = np.isfinite(y) & np.isfinite(yhat)
    y = y[ok]; yhat = yhat[ok]
    mse = mean_squared_error(y, yhat)
    mae = mean_absolute_error(y, yhat)
    ss_res = float(np.sum((yhat - y) ** 2))
    var = float(np.sum((y - y.mean()) ** 2)) if len(y) > 1 else np.nan
    r2 = 1.0 - ss_res / var if var > 0 else np.nan
    pear = float(np.corrcoef(y, yhat)[0, 1]) if len(y) > 1 else np.nan
    ry = pd.Series(y).rank(method="average").to_numpy()
    ryhat = pd.Series(yhat).rank(method="average").to_numpy()
    spear = float(np.corrcoef(ry, ryhat)[0, 1]) if len(y) > 1 else np.nan
    return dict(MSE=mse, MAE=mae, R2=r2, Pearson=pear, Spearman=spear), ok

# -----------------------------
# Main (predict-only)
# -----------------------------
def main():
    torch.backends.cudnn.benchmark = True


    samples = load_dataset2(cfg.DIR.DATASET, cfg.DIR.PREPROCESSED_DIR)
    loader  = DataLoader(samples, batch_size=cfg.MODEL.BATCH_SIZE,
                         shuffle=False, collate_fn=custom_collate_fn2,
                         pin_memory=torch.cuda.is_available())


    odir = cfg.DIR.OUTPUT_DIR
    os.makedirs(odir, exist_ok=True)
    model_path = find_model_path(odir)
    print(f"[Load] model -> {model_path}")

    model = Main_model2(cfg).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)

    t0 = timeit.default_timer()
    y_true, y_pred = predict_all(model, loader)
    t1 = timeit.default_timer()
    print(f"[Predict] N={len(y_true)} in {t1 - t0:.1f}s")

    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    df = pd.DataFrame({
        "label_log10": y_true[ok],
        "pred_log10":  y_pred[ok],
    })
    df["label_S_per_cm"] = np.power(10.0, df["label_log10"])
    df["pred_S_per_cm"]  = np.power(10.0, df["pred_log10"])
    df["residual_log10"] = df["pred_log10"] - df["label_log10"]

    csv_path = os.path.join(odir, "predictions_full.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Saved] CSV -> {csv_path}")

    m, _ = compute_metrics_log10(y_true, y_pred)
    print("[Metrics@log10] "
          f"MSE={m['MSE']:.6f}, MAE={m['MAE']:.6f}, R2={m['R2']:.6f}, "
          f"Pearson={m['Pearson']:.6f}, Spearman={m['Spearman']:.6f}")

if __name__ == "__main__":
    main()
