# MatPES MLIP Force Analysis

A comprehensive benchmarking study of machine learning interatomic potentials (MLIPs) against DFT forces from the [MatPES](https://github.com/materialsvirtuallab/matpes) dataset. This repository contains the analysis notebooks, utility code, and HPC job generators used to produce the results.

---

## Overview

Standard force benchmarks report only the force magnitude error (MAE/RMSE of |ΔF|). This work introduces a complementary metric — the **angular error |Δθ|** — that measures how much the *direction* of the predicted force deviates from the DFT reference. Together, the two metrics give a more complete picture of where MLIPs succeed and where they fail:

| Metric | Symbol | Unit | Captures |
|--------|--------|------|----------|
| Force magnitude error | \|ΔF\| | eV/Å | Error in force strength |
| Force direction error | \|Δθ\| | ° | Error in force direction |

The analysis is performed across **three test sets** and **seven MLIP models**:

### Test Sets

| Dataset | DFT functional | # atoms | Description |
|---------|---------------|---------|-------------|
| MatPES-PBE | PBE | ~6 M | Standard MatPES benchmark |
| MatPES-r2SCAN | r2SCAN | ~6 M | Higher-accuracy r2SCAN references |
| OMAT24 rattled-1000 | PBE | ~300 K | Random displacements of 1 000 OMAT24 structures |

### Models Evaluated

| Model | Architecture | Trained on |
|-------|-------------|------------|
| MACE (MACE-MP-0 medium) | MACE | MPtrj + Alexandria |
| CHGNet | GNN + CHG | MPtrj |
| M3GNet | M3GNet | MP-2021.2.8 |
| UMA (uma-s-1p1) | eSEN | OMAT24 + MPtrj + ODAC |
| M3GNet-MatPES | M3GNet | MatPES-PBE |
| TensorNet-MatPES | TensorNet | MatPES-PBE |
| MACE-MatPES | MACE | MatPES-PBE |

---

## Analysis at a Glance

The notebooks produce a rich set of publication-quality figures:

- **Split-triangle heatmaps** — fraction of atoms with |ΔF| < threshold (lower triangle: also |Δθ| < 1°, upper: |Δθ| < 20°)
- **Cumulative distribution functions** — full CDF of |ΔF| and |Δθ| for all models with zoomed inset
- **2-D density plots** — joint distribution of |ΔF| vs |Δθ|, |ΔF| vs F\_DFT, |Δθ| vs F\_DFT
- **Near-equilibrium / far-from-equilibrium panels** — error statistics split by |F\_DFT| < 1 eV/Å vs ≥ 1 eV/Å
- **MAE/RMSE heatmaps** — conditioned on DFT force magnitude and error thresholds
- **Error histograms** — log-log |ΔF| and linear-log |Δθ| distributions
- **Merged 6-panel overview figure** — ready for publication

---

## Repository Structure

```
MatPES_force_analysis/
│
├── matpes_analysis_mace_matpes.ipynb          # PBE analysis (MatPES + rattled modes)
├── matpes_analysis_r2scan.ipynb               # r2SCAN analysis
├── matpes_analysis_mace_matpes_omat.ipynb     # OMAT24 rattled-1000 analysis
│
├── matpes_run_generator.ipynb                 # HPC job generator — MatPES PBE
├── matpes_r2scan_run_generator.ipynb          # HPC job generator — MatPES r2SCAN
├── matpes_run_generator_omat24_rattled1000.ipynb  # HPC job generator — rattled-1000
│
├── matpes_frac_analysis.py                    # Core analysis functions (fractions, MAE, RMSE, heatmaps)
├── mlip_cdf_density_plots.py                  # CDF and 2-D density plotting utilities
├── heatmap_table.py                           # Low-level heatmap drawing primitives
│
├── atoms_figure.png                           # Crystal structure visualization
├── all_dfs.json                               # Per-structure metadata (32 KB, in repo)
│
├── R2SCAN/                                    # Per-model r2SCAN results (~520 MB each — Git LFS)
│   ├── all_results_mace-mp0_medium_r2SCAN.json
│   ├── all_results_Tensornet_matpes_PBE_r2SCAN.json
│   ├── all_results_mace_matpes_pbe_r2SCAN.json
│   └── all_results_mace_matpes_r2scan_r2SCAN.json
│
├── rattled_1000_results/                      # Per-model rattled-1000 results (~280 MB each — Git LFS)
│   ├── all_results_mace-mp0_medium_PBE.json
│   ├── all_results_chgnet_PBE.json
│   ├── all_results_m3gnet_pes_PBE.json
│   ├── all_results_uma_PBE.json
│   ├── all_results_m3gnet_matpes_pbe_PBE.json
│   ├── all_results_Tensornet_matpes_PBE_PBE.json
│   └── all_results_mace_matpes_pbe_PBE.json
│
├── requirements.txt
├── .gitattributes                             # Git LFS configuration
└── .gitignore
```

---

## Data Access

### Files in the repository (Git LFS)

The `R2SCAN/` and `rattled_1000_results/` JSON files are stored via Git LFS. After cloning, pull them with:

```bash
git lfs pull
```

### Large files hosted externally

Two files exceed the Git LFS 2 GB per-file limit and must be downloaded separately:

| File | Size | Contents |
|------|------|----------|
| `all_results_PBE_all_data.json` | ~2.9 GB | Per-atom force errors for all PBE models |
| `all_results_R2SCAN_all_data.json` | ~2.2 GB | Per-atom force errors for all r2SCAN models |
| `FP_paper_test/all_results_mace_matpes_pbe_PBE.json` | ~661 MB | MACE-MatPES PBE results |

> **Note:** Links to the Zenodo/HuggingFace data record will be added here once the data is deposited.

Place these files in the repository root (and `FP_paper_test/` subdirectory) before running the PBE and r2SCAN analysis notebooks.

---

## Installation

```bash
git clone git@github.com:kamirian/Matpes_analysis.git
cd Matpes_analysis
git lfs pull
pip install -r requirements.txt
```

---

## Running the Analysis Notebooks

All three analysis notebooks assume they are run from the **repository root directory**. No path changes are needed — all paths are relative.

### Notebook 1 — MatPES PBE Analysis

`matpes_analysis_mace_matpes.ipynb`

Evaluates MACE, CHGNet, M3GNet, UMA, M3GNet-MatPES, TensorNet-MatPES, and MACE-MatPES against PBE DFT forces.

**Required files:** `all_dfs.json`, `all_results_PBE_all_data.json`, `FP_paper_test/all_results_mace_matpes_pbe_PBE.json`, `atoms_figure.png`

### Notebook 2 — MatPES r2SCAN Analysis

`matpes_analysis_r2scan.ipynb`

Evaluates the same models (plus r2SCAN-trained variants) against higher-accuracy r2SCAN DFT forces.

**Required files:** `all_dfs.json`, `all_results_R2SCAN_all_data.json`, `R2SCAN/*.json`, `atoms_figure.png`

### Notebook 3 — OMAT24 Rattled-1000 Analysis

`matpes_analysis_mace_matpes_omat.ipynb`

Switch between the MatPES-PBE and rattled-1000 datasets via the `DATASET` variable in Section 0.

**Required files:** `rattled_1000_results/*.json` (for `DATASET = "rattled1000"`), or the PBE files above (for `DATASET = "matpes"`)

---

## HPC Job Generators

The three generator notebooks produce SLURM submission scripts for running MLIP evaluations on a compute cluster. They are not needed to reproduce the analysis — only to re-generate the raw result JSON files.

| Notebook | Target dataset | Output |
|----------|---------------|--------|
| `matpes_run_generator.ipynb` | MatPES-PBE | `all_results_{mlip}_PBE.json` |
| `matpes_r2scan_run_generator.ipynb` | MatPES-r2SCAN | `all_results_{mlip}_r2SCAN.json` |
| `matpes_run_generator_omat24_rattled1000.ipynb` | OMAT24 rattled-1000 | `all_results_{mlip}_PBE.json` |

Each generator follows the same workflow:
1. Chunk the raw dataset into N pieces
2. Generate one Python run script + SLURM file per chunk
3. `rsync` to cluster → `bash mass_submit.sh`
4. Merge per-chunk outputs into a single `all_results_*.json`

**HPC environment requirements:**

| Component | Version / Path |
|-----------|---------------|
| MACE venv | `~/scratch/MACE_env/` |
| CHGNet venv | `~/scratch/chgnet_env/` |
| M3GNet/TensorNet venv | `~/scratch/M3GNet_2/` |
| UMA venv | `~/scratch/Facebook/uma_env/` |
| MatPES-PBE dataset | `MatPES-PBE-2025.1.json.gz` |
| MatPES-r2SCAN dataset | `MatPES-r2SCAN-2025.1.json.gz` |
| rattled-1000 dataset | `rattled-1000.tar.gz` |

---

## Data Format

Each result JSON file has the following structure:

```json
{
  "model_name": {
    "all_F_dft_mags":  [float, ...],   // DFT force magnitudes |F_DFT| (eV/Å), one per atom
    "all_deltaF":      [float, ...],   // Force magnitude errors |ΔF| (eV/Å)
    "all_deltaTheta":  [float, ...],   // Angular errors |Δθ| (degrees)
    "original_indices": [int, ...]     // Structure indices in the source dataset
  }
}
```

---

## Citation

If you use this code or analysis in your work, please cite:

```bibtex
@misc{amirian2025matpes,
  author = {Kiyana Amirian},
  title  = {MatPES MLIP Force Analysis},
  year   = {2025},
  url    = {https://github.com/kamirian/Matpes_analysis}
}
```

---

## License

MIT
