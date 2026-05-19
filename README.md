# MatPES MLIP Force Analysis

A systematic study of force prediction accuracy in machine learning interatomic potentials (MLIPs), evaluated against DFT forces from the [MatPES](https://github.com/materialsvirtuallab/matpes) dataset. The central argument: **MAE/RMSE of force magnitude alone is an incomplete benchmark** — it masks failure modes that matter in practice.

---

## Motivation

Current MLIP evaluations overwhelmingly focus on average force magnitude error (MAE/RMSE of |ΔF|). This captures only part of the picture. A model can achieve low MAE while still:

- Predicting forces that point in completely wrong directions (high angular error |Δθ|)
- Producing catastrophically large errors on a small but consequential fraction of atoms
- Performing well on near-equilibrium atoms while failing on far-from-equilibrium configurations

These failures matter because **force accuracy requirements in practice are strict**. Static relaxations and NEB calculations typically require force convergence of 0.001–0.01 eV/Å. Large or misdirected force errors during molecular dynamics can lead to unphysical atomic trajectories and simulation crashes.

This work addresses the gap by decomposing MLIP force accuracy across **three regimes**:

| Regime | Definition | Why it matters |
|--------|-----------|----------------|
| DFT-substitutable | \|ΔF\| < 0.01–0.1 eV/Å and \|Δθ\| < 1–20° | Required for converged relaxations and reliable NEB barriers |
| Catastrophic failures | \|ΔF\| > 1–10 eV/Å | Can destabilize MD trajectories; rarely reported by average metrics |
| Far-from-equilibrium | \|F\_DFT\| > 1 eV/Å | ~26.8% of MatPES atoms vs ~6.1% of MPtrj — most benchmarks underrepresent this regime |

**Key findings:**

- For most current MLIPs, jointly accurate predictions (both low |ΔF| **and** low |Δθ|) are rare
- The **fraction of low-force-error atoms** is a more informative metric than MAE/RMSE alone
- The **fraction of large-error atoms** is a distinct, equally important metric — catastrophic failures are hidden by averaging
- Whether a model handles far-from-equilibrium atoms well depends heavily on whether the training data included them — the MatPES dataset was explicitly designed to sample these configurations

---

## Metrics

| Metric | Symbol | Unit | What it captures |
|--------|--------|------|-----------------|
| Force magnitude error | \|ΔF\| | eV/Å | How far off is the predicted force *strength*? |
| Force direction error | \|Δθ\| | degrees | How wrong is the predicted force *direction*? |

These two metrics are complementary — a model can fail on one while appearing fine on the other. Analyzing their joint distribution reveals failure modes that neither metric alone can expose.

---

## Test Sets

| Dataset | Functional | # atoms | Description |
|---------|-----------|---------|-------------|
| MatPES-PBE | PBE | ~6 M | Main benchmark; 26.8% far-from-equilibrium atoms |
| MatPES-r2SCAN | r2SCAN | ~6 M | Higher-accuracy references; same structures |
| OMAT24 rattled-1000 | PBE | ~300 K | 1 000 OMAT24 structures with random displacements |

## Models Evaluated

| Model | Architecture | Training set |
|-------|-------------|-------------|
| MACE (MACE-MP-0 medium) | MACE | MPtrj + Alexandria |
| CHGNet | GNN + charge | MPtrj |
| M3GNet | M3GNet | MP-2021.2.8 |
| UMA (uma-s-1p1) | eSEN | OMAT24 + MPtrj + ODAC |
| M3GNet-MatPES | M3GNet | MatPES-PBE |
| TensorNet-MatPES | TensorNet | MatPES-PBE |
| MACE-MatPES | MACE | MatPES-PBE |

The r2SCAN notebook also evaluates `mace_matpes_r2scan`, `m3gnet_matpes_r2scan`, and `TensorNET_matpes_R2SCAN` — variants fine-tuned on the r2SCAN functional.

---

## Analysis at a Glance

The notebooks produce a rich set of publication-quality figures:

- **Split-triangle heatmaps** — fraction of atoms with |ΔF| < threshold (lower triangle: also |Δθ| < 1°, upper: |Δθ| < 20°), showing joint magnitude-angle accuracy
- **Cumulative distribution functions** — full CDF of |ΔF| and |Δθ| with zoomed inset on the high-accuracy tail
- **2-D density plots** — joint distributions of |ΔF| vs |Δθ|, |ΔF| vs F\_DFT, and |Δθ| vs F\_DFT
- **Near-eq / far-from-eq regime panels** — error statistics split by |F\_DFT| < 1 eV/Å vs ≥ 1 eV/Å
- **Large-error fraction heatmaps** — fraction of atoms with |ΔF| > threshold (log-scale), the catastrophic failure metric
- **MAE/RMSE heatmaps** — triangular heatmaps conditioned on |F\_DFT| thresholds (lower = MAE, upper = RMSE)
- **Error histograms** — log-log |ΔF| and linear-log |Δθ| distributions, all atoms and near-equilibrium subset
- **Merged 6-panel overview figure** — crystal structure + density plots + CDFs, ready for publication

---

## Repository Structure

```
MatPES_force_analysis/
│
│  ── Analysis notebooks ──
├── matpes_analysis_mace_matpes.ipynb          # PBE analysis: MatPES & rattled-1000
├── matpes_analysis_r2scan.ipynb               # r2SCAN analysis
├── matpes_analysis_mace_matpes_omat.ipynb     # OMAT24 rattled-1000 analysis
│
│  ── HPC job generators (re-run evaluations on cluster) ──
├── matpes_run_generator.ipynb                 # MatPES-PBE SLURM generator
├── matpes_r2scan_run_generator.ipynb          # MatPES-r2SCAN SLURM generator
├── matpes_run_generator_omat24_rattled1000.ipynb  # Rattled-1000 SLURM generator
│
│  ── Python utility modules ──
├── matpes_frac_analysis.py     # Core analysis: fraction tables, MAE/RMSE, regime panels, heatmaps
├── mlip_cdf_density_plots.py   # CDF computation and 2-D density plotting
├── heatmap_table.py            # Low-level heatmap drawing primitives (triangular cells, colorbars)
│
│  ── Data ──
├── atoms_figure.png                           # Crystal structure visualization
├── all_dfs.json                               # Per-structure metadata (32 KB)
├── R2SCAN/                                    # Per-model r2SCAN results (~520 MB each — Git LFS)
└── rattled_1000_results/                      # Per-model rattled-1000 results (~280 MB each — Git LFS)
```

---

## Python Modules

### `matpes_frac_analysis.py` — Core analysis engine

The computational backbone of all three analysis notebooks. Provides:

- **Fraction tables** — `build_frac_table`, `build_dF_frac_table`: compute the fraction of atoms with |ΔF| below threshold, optionally also requiring |Δθ| < cut
- **Regime panels** — `build_regime_panels`, `build_theta_regime_panels`: split atoms into near-equilibrium (|F\_DFT| < 1 eV/Å) and far-from-equilibrium (|F\_DFT| ≥ 1 eV/Å) and compute error statistics for each
- **Conditioned MAE/RMSE** — `build_F_dft_conditioned_mae_rmse`, `build_theta_conditioned_mae_rmse`: error statistics as a function of |F\_DFT| or |Δθ| thresholds
- **Large-error analysis** — `build_dF_frac_larger_table`, `build_large_dF_fdft_table`: fraction of atoms with |ΔF| *above* threshold and the |F\_DFT| distribution of those catastrophic-error atoms
- **Visualization** — `split_triangle_heatmap`, `single_heatmap`, `plot_error_histograms`, `plot_fraction_panel`: all figure types used in the notebooks
- **Querying** — `get_bad_atom_indices`, `get_bad_structure_indices`: find atoms/structures meeting any combination of |ΔF|, |Δθ|, and |F\_DFT| criteria

### `mlip_cdf_density_plots.py` — CDF and density visualization

Handles the continuous distribution figures:

- **CDF computation** — `build_cdf_from_all_results`, `build_cdf_by_regime`: compute cumulative distributions of |ΔF| and |Δθ| for all models, with optional near-eq / far-from-eq split
- **2-D density panels** — `panel_abs_dF_vs_dtheta_cond_on_Fdft`, `panel_Fdft_vs_abs_dF`, `panel_Fdft_vs_dtheta`: joint density heatmaps with marginal histograms, PCHIP-smoothed CDFs
- **CDF plots** — `plot_cdf_with_inset_on_ax`: full CDF with automatically placed inset zoom for the high-accuracy tail or model-separation region

### `heatmap_table.py` — Heatmap drawing primitives

Low-level matplotlib code for the custom split-triangle heatmap style used throughout the paper. Provides `draw_triangular_cell`, `draw_rectangular_column`, `add_colorbar`, and the high-level `triangular_heatmap_with_fraction_row` that assembles a complete publication-style heatmap table.

---

## Data Access

### Files tracked by Git LFS

```bash
git lfs pull   # download R2SCAN/ and rattled_1000_results/ after cloning
```

### Large files hosted externally (> 2 GB — beyond Git LFS limit)

| File | Size | Contents |
|------|------|----------|
| `all_results_PBE_all_data.json` | ~2.9 GB | Per-atom force errors, all PBE models |
| `all_results_R2SCAN_all_data.json` | ~2.2 GB | Per-atom force errors, all r2SCAN models |
| `FP_paper_test/all_results_mace_matpes_pbe_PBE.json` | ~661 MB | MACE-MatPES full PBE results |

> Download links will be added here once the data is deposited on Zenodo / HuggingFace Datasets.

---

## Installation & Quick Start

```bash
git clone https://github.com/kamirian/Matpes_analysis.git
cd Matpes_analysis
git lfs pull
pip install -r requirements.txt

# Place the externally-hosted JSON files in the repo root, then:
jupyter notebook matpes_analysis_mace_matpes.ipynb
```

All notebooks use relative paths — no path changes needed as long as you run from the repo root.

---

## HPC Job Generators

The generator notebooks produce SLURM scripts for re-running MLIP evaluations on a cluster. They are **not needed to reproduce the analysis** — only to regenerate raw result JSON files.

| Notebook | Dataset | Output |
|----------|---------|--------|
| `matpes_run_generator.ipynb` | MatPES-PBE | `all_results_{mlip}_PBE.json` |
| `matpes_r2scan_run_generator.ipynb` | MatPES-r2SCAN | `all_results_{mlip}_r2SCAN.json` |
| `matpes_run_generator_omat24_rattled1000.ipynb` | Rattled-1000 | `all_results_{mlip}_PBE.json` |

Workflow: chunk dataset → generate scripts → `rsync` to cluster → `bash mass_submit.sh` → merge outputs.

---

## Data Format

```json
{
  "model_name": {
    "all_F_dft_mags":   [...],  // DFT force magnitudes |F_DFT| (eV/Å)
    "all_deltaF":       [...],  // Force magnitude errors |ΔF| (eV/Å)
    "all_deltaTheta":   [...],  // Angular errors |Δθ| (degrees)
    "original_indices": [...]   // Structure indices in the source dataset
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
