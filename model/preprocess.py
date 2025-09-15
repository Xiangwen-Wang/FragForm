import os
import re
import pickle
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from rdkit import Chem

from configs2 import get_cfg_defaults

# =============================

cfg = get_cfg_defaults()

NUM_ATOM_TYPES = cfg.MODEL.NUM_ATOM_TYPES
RADIUS = cfg.SUB.RADIUS
CSV_PATH = cfg.DIR.DATASET
OUT_DIR = cfg.DIR.PREPROCESSED_DIR


# =============================

subgraph_dict: Dict = defaultdict(lambda: len(subgraph_dict))
atom_dict: Dict = defaultdict(lambda: len(atom_dict))
bond_dict: Dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict: Dict = defaultdict(lambda: len(fingerprint_dict))
edge_dict: Dict = defaultdict(lambda: len(edge_dict))

# =============================


def create_atom_features(mol: Chem.Mol) -> np.ndarray:
    syms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    for atom in mol.GetAromaticAtoms():
        idx = atom.GetIdx()
        syms[idx] = (syms[idx], 'aromatic')
    feats: List[int] = []
    for s in syms:
        idx = atom_dict[s]
        if idx >= NUM_ATOM_TYPES:
            print(f" Atom type {s} assigned index {idx}, exceeding NUM_ATOM_TYPES ({NUM_ATOM_TYPES}). Using default index 0.")
            idx = 0
        feats.append(idx)
    return np.array(feats, dtype=np.int64)


def create_ijbonddict(mol: Chem.Mol):
    i_jbond_dict = defaultdict(list)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond_dict[str(bond.GetBondType())]
        i_jbond_dict[i].append((j, bond_type))
        i_jbond_dict[j].append((i, bond_type))
    return i_jbond_dict


def extract_fingerprints(atoms: np.ndarray, i_jbond_dict: dict, radius: int) -> np.ndarray:
    nodes = list(atoms.tolist())
    for _ in range(radius):
        new_nodes = []
        for i in range(len(nodes)):
            neighbors = i_jbond_dict.get(i, [])
            labels = [(nodes[i],)]
            for (j, b) in neighbors:
                if 0 <= j < len(nodes):
                    labels.append((nodes[j], b))
            labels = tuple(sorted(labels))
            new_id = subgraph_dict.setdefault(labels, len(subgraph_dict))
            new_nodes.append(new_id)
        nodes = new_nodes
    return np.array(nodes, dtype=np.int64)


def create_bond_adjacency(mol: Chem.Mol) -> np.ndarray:
    n = mol.GetNumAtoms()
    adj = np.zeros((n, n), dtype=np.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond_dict[str(bond.GetBondType())]
        adj[i, j] = adj[j, i] = bt
    return adj


def get_heavy_atom_indices(mol: Chem.Mol):
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]


def build_full_adjacency(mol: Chem.Mol):
    n = mol.GetNumAtoms()
    adj = np.zeros((n, n), dtype=np.int8)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[i, j] = 1
        adj[j, i] = 1
    return adj


def k_hop_neighbors(adj_bin: np.ndarray, seeds: np.ndarray, k: int):
    reachable = np.zeros(adj_bin.shape[0], dtype=bool)
    frontier = np.zeros_like(reachable)
    reachable[seeds] = True
    frontier[seeds] = True
    for _ in range(k):
        nxt = adj_bin.dot(frontier.astype(np.int8)) > 0
        frontier = np.logical_and(nxt, ~reachable)
        reachable = np.logical_or(reachable, frontier)
        if not frontier.any():
            break
    return reachable


def make_subgraph_masks_per_heavy_atom(mol, include_hydrogens=False):
    N = mol.GetNumAtoms()
    heavy = np.array(get_heavy_atom_indices(mol), dtype=np.int64)
    if len(heavy) == 0:
        return np.zeros((0, N), dtype=bool)

    adj_full = build_full_adjacency(mol)

    mask_heavy = np.zeros(N, dtype=bool)
    mask_heavy[heavy] = True
    adj_heavy = adj_full.copy()
    adj_heavy[:, ~mask_heavy] = 0
    adj_heavy[~mask_heavy, :] = 0

    masks = []
    for h in heavy:
        seed = np.array([h], dtype=np.int64)
        heavy_reach = k_hop_neighbors(adj_heavy, seed, k=1)  # 固定 1-hop
        if include_hydrogens:
            include = heavy_reach.copy()
            heavy_idxs = np.where(heavy_reach)[0]
            hydro_mask = ~mask_heavy
            neigh_of_heavy = (adj_full[heavy_idxs].sum(axis=0) > 0)
            add_h = np.logical_and(neigh_of_heavy, hydro_mask)
            include = np.logical_or(include, add_h)
            masks.append(include)
        else:
            masks.append(heavy_reach)

    return np.stack(masks, axis=0)

# =============================


def load_table(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep='\t')
    except Exception:
        df = pd.read_csv(path, sep='\t')
    norm = {c: re.sub(r"\s+", "", str(c).lower()) for c in df.columns}
    df = df.rename(columns=norm)
    return df

# =============================

def collect_unique_substrate_smiles(df: pd.DataFrame) -> List[str]:
    sub_cols = [f'substrate{i}' for i in range(1, 6) if f'substrate{i}' in df.columns]
    con_cols = [f'con{i}' for i in range(1, 6) if f'con{i}' in df.columns]

    pairs = []
    for i in range(1, 6):
        s_col, c_col = f'substrate{i}', f'con{i}'
        if s_col in df.columns and c_col in df.columns:
            pairs.append((s_col, c_col))

    unique_smiles = set()
    for s_col, c_col in pairs:
        sub = df[[s_col, c_col]].dropna()
        con_vals = pd.to_numeric(sub[c_col], errors='coerce')
        sub = sub[con_vals > 0]
        smiles_list = (
            sub[s_col]
            .astype(str)
            .str.strip()
            .replace({'': np.nan, 'nan': np.nan, 'none': np.nan, 'O': np.nan})
            .dropna()
            .tolist()
        )
        unique_smiles.update(smiles_list)

    return sorted(unique_smiles)

# =============================
# Main

def preprocess_substrates(csv_path: str, output_dir: str, radius: int):


    df = load_table(csv_path)
    print(f"Found {df.shape[0]} rows in dataset: {csv_path}")

    smiles_list = collect_unique_substrate_smiles(df)
    print(f"Collected {len(smiles_list)} unique substrate SMILES with con>0.")

    results = {}

    for smiles in smiles_list:
        s = smiles.strip()
        if s.lower() in ('nan', 'none', ''):
            continue

        mol = Chem.MolFromSmiles(s)
        if mol is None:
            print(f"Invalid SMILES: {s}, skipping.")
            continue

        mol = Chem.AddHs(mol)

        atom_features = create_atom_features(mol)
        adjacency = create_bond_adjacency(mol)
        i_jbond = create_ijbonddict(mol)

        try:
            fingerprints = extract_fingerprints(atom_features, i_jbond, radius)
        except Exception as e:
            print(f"  Fingerprint error for {s}: {e} -> fallback radius=0")
            fingerprints = extract_fingerprints(atom_features, i_jbond, 0)

        subgraph_masks = make_subgraph_masks_per_heavy_atom(mol, include_hydrogens=False)

        results[s] = {
            'smiles': s,
            'atom_features': atom_features,
            'adjacency': adjacency,
            'fingerprints': fingerprints,
            'subgraph_masks': subgraph_masks,
        }

    with open(output_dir, 'wb') as f:
        pickle.dump({
            'results': results,
            'atom_dict': dict(atom_dict),
            'bond_dict': dict(bond_dict),
            'fingerprint_dict': dict(fingerprint_dict),
            'edge_dict': dict(edge_dict),
            'num_atom_types': len(atom_dict)
        }, f)

    print(f"Processed {len(results)} molecules. Results saved to {output_dir}")
    return results


if __name__ == '__main__':
    preprocess_substrates(CSV_PATH, OUT_DIR, RADIUS)
