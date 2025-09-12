# FragForm: Fragment-level Formulation Design

This repository contains the code, datasets, and results associated with the paper:  
**"Fragment- and composition-aware deep learning for electrolyte formulation design"**

FragForm is a fragment-attribution and composition-aware deep learning framework that links molecular substructures to mixture-level properties, with application to electrolyte conductivity prediction.

---

## Repository Structure

- **`model/`**  
  Contains all model code and training scripts.  
  - `train_randomdiv.py` – main training entry point  
  - `evaluate.py` – model evaluation on test sets  
  - `configs.py` – configuration files (hyperparameters, dataset paths, training options)  
  - `model.py` – implementation of GNN layers, attention, and attribution modules
  - `baseline_model.py` – implementation of simplified baseline modules

- **`dataset/`**  
  Contains electrolyte formulation datasets.  
  - `MolSets` – benchmark dataset with binary mixtures and fixed ratios  
  - `SMI-TED` – large-scale dataset with explicit compositions  
  - `combined` – cleaned and aligned merged dataset  
  Each dataset includes SMILES strings, composition ratios, and measured ionic conductivities.  

- **`results/`**  
  Contains trained model outputs, figures, and evaluation results.  
  - `regression.png` – predicted vs. experimental conductivity regression plots  
  - `FG.png` – functional group attribution and pairwise interactions  
  - `overview.png` – model architecture and workflow  
  - `dataset.png` – dataset comparison and analysis  

---

##  Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/FragForm.git
cd FragForm
```

### 2. Create environment
We recommend Python 3.9+ with PyTorch installed.
Install dependencies:
```bash
pip install -r requirements.txt
```
### 3. Train the model
```bash
python model/train_randomdiv.py
```
### 4. Evaluate on test set
```bash
python model/evaluate.py
```
### 5. Attribution analysis
```bash
python model/analyze.py
```
