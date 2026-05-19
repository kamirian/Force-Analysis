"""
matpes_frac_analysis.py
=======================
Analysis and visualization helpers for MatPES MLIP force/energy evaluation.

Public API
----------
Computation
  frac_percent(dF, dtheta, F_dft, dF_cut, angle_cut, fdf_min)
      → fraction (%) of atoms passing |ΔF| < dF_cut AND Δθ < angle_cut

  build_frac_table(all_results, dF_cuts, angle_cuts, fdf_min)
      → {angle_cut: DataFrame(index=models, columns=dF_cuts)}

  build_dF_frac_table(all_results, dF_thresholds, fdf_min)
      → DataFrame(index=models, columns=thresholds)  — fraction < threshold, no angle filter

  build_dF_frac_larger_table(all_results, dF_thresholds, fdf_min)
      → DataFrame(index=models, columns=thresholds)  — fraction > threshold (large errors)

  build_regime_panels(all_results, abs_thresh, rel_thresh, threshold, eps, fdf_min)
      → (df_panel_A, df_panel_B)  — near-eq / far-from-eq atom statistics

  build_large_dF_fdft_table(all_results, df_thresh, fdft_thresh, fdf_min)
      → DataFrame  — F_DFT distribution for atoms with |ΔF| > df_thresh

  merge_mae_rmse_as_string(mae_df, rmse_df, fmt)
      → DataFrame  — cells formatted as "mae / rmse"

  build_F_dft_frac_table(all_results, fdft_thresholds)
      → DataFrame  — fraction (%) of atoms with |F_DFT| > threshold

  build_F_dft_conditioned_mae_rmse(all_results, fdft_thresholds)
      → (mae_df, rmse_df)  — MAE/RMSE of |ΔF| conditioned on |F_DFT| > threshold

  build_theta_frac_table(all_results, theta_thresholds, fdf_min)
      → DataFrame  — fraction (%) of atoms with |Δθ| < threshold

  build_theta_conditioned_mae_rmse(all_results, theta_thresholds, fdf_min)
      → (mae_df, rmse_df)  — MAE/RMSE of |Δθ| conditioned on |Δθ| < threshold

  build_theta_regime_panels(all_results, theta_thresh, threshold, fdf_min)
      → (df_panel_A, df_panel_B)  — near-eq / far-from-eq Δθ fraction statistics

Plotting
  split_triangle_heatmap(data_lower, data_upper, row_labels, col_labels, ...)
      → fig, ax  — each cell split diagonally: lower ← data_lower, upper ← data_upper

  single_heatmap(data, row_labels, col_labels, ...)
      → fig, ax  — standard rectangular heatmap, one value per cell

  merged_heatmaps(data_left, ..., data_lower, data_upper, ...)
      → fig, (ax_left, ax_right)  — single_heatmap + split_triangle side-by-side

  heatmap_fraction_delta_theta(df_frac, ...)
      → fig, ax  — imshow-based heatmap for Δθ fraction data

  triangular_heatmap_with_fraction_row_word_style(mae_df, rmse_df, frac_row_str, ...)
      → fig, ax  — triangular heatmap with horizontal colorbars placed below

  plot_error_histograms(all_results_filtered, ...)
      — |ΔF| (log-x, log-y) and |Δθ| (linear-x, log-y) histograms
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon, Rectangle
from matplotlib.colors import Normalize


# ── global matplotlib style (matches tested figure output) ────────────────────
plt.rcParams["svg.fonttype"]       = "none"
plt.rcParams["font.family"]        = "Arial"
plt.rcParams["mathtext.rm"]        = "Arial"
plt.rcParams["mathtext.it"]        = "Arial:italic"
plt.rcParams["mathtext.fontset"]   = "custom"
plt.rcParams["mathtext.default"]   = "it"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _sigfig_formatter(n):
    """
    Return a callable that formats a float to *n* significant figures as plain
    decimal (never scientific notation).  E.g. n=2: 99.5→"100", 7.3→"7.3",
    0.12→"0.12".
    """
    import math
    def _f(val):
        if not np.isfinite(val):
            return ""
        if val == 0:
            return "0"
        s = f"{val:.{n}g}"
        if "e" in s or "E" in s:
            mag = math.floor(math.log10(abs(val)))
            dp  = max(0, n - 1 - mag)
            s   = f"{val:.{dp}f}"
        return s
    return _f


def _fmtval(fmt, val):
    """Apply *fmt* to *val*: callable → fmt(val), str → fmt.format(val)."""
    return fmt(val) if callable(fmt) else fmt.format(val)


def _text_color(rgba, threshold=0.5):
    """Return 'black' or 'white' based on WCAG background luminance."""
    r, g, b, _ = rgba
    return "black" if (0.2126 * r + 0.7152 * g + 0.0722 * b) > threshold else "white"


# ═════════════════════════════════════════════════════════════════════════════
# Computation
# ═════════════════════════════════════════════════════════════════════════════

def frac_percent(dF, dtheta, F_dft, dF_cut, angle_cut, fdf_min=0.01):
    """
    Fraction (%) of atoms satisfying both thresholds simultaneously.

    Only atoms with |F_dft| > fdf_min are considered (near-zero DFT forces are
    excluded from both numerator and denominator).

    Parameters
    ----------
    dF        : array-like – force magnitude error |F_mlip| − |F_dft| (eV/Å)
    dtheta    : array-like – angular error between force vectors (degrees)
    F_dft     : array-like – DFT force magnitude (eV/Å)
    dF_cut    : float – threshold on |ΔF| (eV/Å)
    angle_cut : float – threshold on Δθ (degrees); use 180 to disable angle filter
    fdf_min   : float – minimum |F_dft| for an atom to be included

    Returns
    -------
    float in [0, 100], or np.nan if no valid atoms exist
    """
    F_dft  = np.asarray(F_dft,  float)
    dF     = np.asarray(dF,     float)
    dtheta = np.asarray(dtheta, float)

    base = np.isfinite(F_dft) & np.isfinite(dF) & np.isfinite(dtheta)
    base &= np.abs(F_dft) > fdf_min

    denom = base.sum()
    if denom == 0:
        return np.nan

    ok = base & (np.abs(dF) < dF_cut) & (np.abs(dtheta) < angle_cut)
    return 100.0 * ok.sum() / denom


def build_frac_table(all_results, dF_cuts, angle_cuts, fdf_min=0.01):
    """
    Compute fraction tables for one or more angle-cut thresholds.

    Parameters
    ----------
    all_results : dict[model_name → dict]
        Each value must have keys "F_dft", "deltaF", "deltaTheta".
    dF_cuts     : list of float – force-error thresholds (eV/Å)
    angle_cuts  : list of float – angular thresholds (degrees), e.g. [1, 20]
    fdf_min     : float – minimum |F_dft| for inclusion

    Returns
    -------
    dict[angle_cut → pd.DataFrame]
        Rows = models, columns = dF_cuts, values = fraction (%) of atoms
        passing |ΔF| < dF_cut AND Δθ < angle_cut.
    """
    tables = {}
    for angle_cut in angle_cuts:
        rows = {}
        for model, data in all_results.items():
            F_dft  = data.get("F_dft", data.get("all_F_dft_mags"))
            dF     = data.get("deltaF", data.get("all_deltaF"))
            dtheta = data.get("deltaTheta", data.get("all_deltaTheta"))
            rows[model] = {
                thr: frac_percent(dF, dtheta, F_dft, thr, angle_cut, fdf_min)
                for thr in dF_cuts
            }
        df = pd.DataFrame(rows).T          # rows = models, columns = thresholds
        df.columns = dF_cuts
        tables[angle_cut] = df
    return tables


def build_dF_frac_table(all_results, dF_thresholds, fdf_min=0.01):
    """
    Fraction (%) of atoms with |ΔF| < threshold — no angle filter.

    Parameters
    ----------
    all_results   : dict[model_name → dict]  must have keys "F_dft", "deltaF"
    dF_thresholds : list of float
    fdf_min       : float

    Returns
    -------
    pd.DataFrame – rows = models, columns = thresholds, values = fraction (%)
    """
    rows = {}
    for model, data in all_results.items():
        F_dft = np.asarray(data.get("F_dft", data.get("all_F_dft_mags")), float)
        dF    = np.asarray(data.get("deltaF", data.get("all_deltaF")),    float)

        mask = np.isfinite(F_dft) & np.isfinite(dF) & (np.abs(F_dft) > fdf_min)
        dF_f = dF[mask]

        rows[model] = {
            thr: (np.round(np.mean(np.abs(dF_f) < thr) * 100, 2) if dF_f.size > 0 else np.nan)
            for thr in dF_thresholds
        }
    df = pd.DataFrame(rows).T
    df.columns = dF_thresholds
    return df


def build_dF_frac_larger_table(all_results, dF_thresholds, fdf_min=0.01):
    """
    Fraction (%) of atoms with |ΔF| **greater than** threshold — no angle filter.

    Complement of build_dF_frac_table; useful for identifying the tail of large
    force errors (e.g. how many atoms have |ΔF| > 1, > 5, > 10 eV/Å).

    Parameters
    ----------
    all_results   : dict[model_name → dict]  must have keys "F_dft", "deltaF"
    dF_thresholds : list of float
    fdf_min       : float – minimum |F_dft| for inclusion

    Returns
    -------
    pd.DataFrame – rows = models, columns = thresholds, values = fraction (%)
    """
    rows = {}
    for model, data in all_results.items():
        F_dft = np.asarray(data.get("F_dft", data.get("all_F_dft_mags")),  float)
        dF    = np.asarray(data.get("deltaF", data.get("all_deltaF")), float)

        mask = np.isfinite(F_dft) & np.isfinite(dF) & (np.abs(F_dft) > fdf_min)
        dF_f = dF[mask]

        rows[model] = {
            thr: (np.round(np.mean(np.abs(dF_f) > thr) * 100, 4) if dF_f.size > 0 else np.nan)
            for thr in dF_thresholds
        }
    df = pd.DataFrame(rows).T
    df.columns = dF_thresholds
    return df


def build_regime_panels(
    all_results,
    abs_thresh,
    rel_thresh,
    threshold=1.0,
    eps=0.01,
    fdf_min=0.0,
):
    """
    Split atoms into two regimes by |F_DFT| and compute absolute/relative
    force-error fraction tables.

    Panel A — near-equilibrium   : |F_DFT| < threshold
    Panel B — far-from-equilibrium: |F_DFT| ≥ threshold

    Each row in the returned DataFrames contains:
      "N atoms"               — number of atoms in this regime for this model
      "Frac of all atoms [%]" — share of the total valid atom pool
      "|ΔF| < {thr} [%]"     — for each absolute threshold
      "r < {thr} [%]"        — for each relative threshold (r = |ΔF| / (|F_DFT| + eps))

    Parameters
    ----------
    all_results : dict[model_name → dict]  keys: "F_dft", "deltaF"
    abs_thresh  : list of float – absolute |ΔF| thresholds (eV/Å)
    rel_thresh  : list of float – relative error thresholds (dimensionless)
    threshold   : float – |F_DFT| boundary between panels (default 1.0 eV/Å)
    eps         : float – stabiliser in relative error denominator
    fdf_min     : float – minimum |F_dft| for an atom to be included at all

    Returns
    -------
    (df_panel_A, df_panel_B) : two pd.DataFrames with rows = models
    """
    panel_A, panel_B = {}, {}

    for model, data in all_results.items():
        F_dft = np.asarray(data.get("F_dft", data.get("all_F_dft_mags")),  float)
        dF    = np.asarray(data.get("deltaF", data.get("all_deltaF")), float)

        n = min(len(F_dft), len(dF))
        F_dft, dF = F_dft[:n], dF[:n]

        valid = np.isfinite(F_dft) & np.isfinite(dF)
        if fdf_min > 0:
            valid &= np.abs(F_dft) > fdf_min

        F_abs  = np.abs(F_dft[valid])
        dF_abs = np.abs(dF[valid])
        rel    = dF_abs / (F_abs + eps)
        total  = F_abs.size

        for panel_dict, mask in [
            (panel_A, F_abs <  threshold),
            (panel_B, F_abs >= threshold),
        ]:
            n_regime = int(mask.sum())
            row = {
                "N atoms":               n_regime,
                "Frac of all atoms [%]": round(100 * n_regime / total, 4) if total > 0 else np.nan,
            }
            for thr in abs_thresh:
                row[f"|ΔF| < {thr} [%]"] = (
                    round(100 * np.mean(dF_abs[mask] < thr), 4) if n_regime > 0 else np.nan
                )
            for thr in rel_thresh:
                row[f"r < {thr} [%]"] = (
                    round(100 * np.mean(rel[mask] < thr), 4) if n_regime > 0 else np.nan
                )
            panel_dict[model] = row

    return pd.DataFrame(panel_A).T, pd.DataFrame(panel_B).T


def build_large_dF_fdft_table(all_results, df_thresh, fdft_thresh, fdf_min=0.01):
    """
    For atoms with |ΔF| > df_thresh, show what fraction have |F_DFT| below
    each of the given thresholds.

    Useful for understanding whether large force errors occur predominantly
    in near-equilibrium (low F_DFT) or high-force atoms.

    Parameters
    ----------
    all_results  : dict[model_name → dict]  keys: "F_dft", "deltaF"
    df_thresh    : float – |ΔF| threshold defining "large-error" atoms (eV/Å)
    fdft_thresh  : list of float – |F_DFT| thresholds to report (eV/Å)
    fdf_min      : float – minimum |F_dft| for an atom to be included

    Returns
    -------
    pd.DataFrame
        Rows = models.
        First column  : "% of all atoms (|ΔF|>{df_thresh:g})"
        Other columns : "|F_DFT| < {thr} [%]" for each fdft_thresh value.
    """
    first_col = f"% of all atoms\n(|ΔF|>{df_thresh:g})"
    rows = {}

    for model, data in all_results.items():
        F_dft = np.abs(np.asarray(data.get("F_dft", data.get("all_F_dft_mags")),  float))
        dF    = np.asarray(data.get("deltaF", data.get("all_deltaF")), float)

        n = min(len(F_dft), len(dF))
        F_dft, dF = F_dft[:n], dF[:n]

        valid  = np.isfinite(F_dft) & np.isfinite(dF) & (F_dft > fdf_min)
        F_dft  = F_dft[valid]
        abs_dF = np.abs(dF[valid])
        total  = int(valid.sum())

        mask_large = abs_dF > df_thresh
        n_large    = int(mask_large.sum())

        row = {first_col: round(100 * n_large / total, 4) if total > 0 else np.nan}
        for thr in fdft_thresh:
            row[f"|F_DFT| < {thr} [%]"] = (
                round(100 * np.mean(F_dft[mask_large] < thr), 2) if n_large > 0 else np.nan
            )
        rows[model] = row

    return pd.DataFrame(rows).T


# ═════════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════════

def split_triangle_heatmap(
    data_lower, data_upper,
    row_labels, col_labels,
    title=None,
    cmap_lower="viridis",
    cmap_upper="viridis",
    norm_lower=None,
    norm_upper=None,
    annotate=True,
    fmt="{:.1f}",
    sig_figs=None,             # significant figures: overrides fmt (e.g. sig_figs=2 → 2 sig figs)
    textsize=8,
    addsize=1,
    cbar=True,
    cbar_label_lower="[%] (<1°)",
    cbar_label_upper="[%] (<20°)",
    figsize=(3.5, 5.5),
    gap=0.2,
    cbar_width=0.015,
    gap_between_cbars=0.15,
    left=0.22, right=0.80, bottom=0.0225, top=1.0,
    savepath=None,
    show_row_labels=False,
):
    """
    Heatmap where each cell is split diagonally into two triangles.

    Lower triangle  ← data_lower  (e.g. fraction at Δθ < 1°)
    Upper triangle  ← data_upper  (e.g. fraction at Δθ < 20°)

    All layout parameters (figsize, margins, font sizes) match tested output.

    Parameters
    ----------
    data_lower, data_upper : 2D array-like, shape (n_models, n_cols)
    row_labels  : list of str – model names (top → bottom)
    col_labels  : list of str – column header labels
    cmap_lower, cmap_upper : str – Matplotlib colormap names
    norm_lower, norm_upper : Normalize or None – auto-computed if None
    fmt         : format string for annotations (applied to both triangles)
    textsize    : int – base font size for annotations
    addsize     : int – added to textsize for tick labels and title
    gap         : float – gap from heatmap right edge to first colorbar
    cbar_width  : float – width of each colorbar (figure-fraction units)
    gap_between_cbars : float
    left, right, bottom, top : float – subplots_adjust margins
    savepath    : str or None – save path (e.g. "out.svg"); None = don't save

    Returns
    -------
    fig, ax
    """
    if sig_figs is not None:
        fmt = _sigfig_formatter(sig_figs)
    data_lower = np.asarray(data_lower, float)
    data_upper = np.asarray(data_upper, float)
    assert data_lower.shape == data_upper.shape, "data_lower and data_upper must have the same shape"
    nrows, ncols = data_lower.shape

    if norm_lower is None:
        v = data_lower[np.isfinite(data_lower)]
        norm_lower = Normalize(vmin=v.min(), vmax=v.max()) if v.size else Normalize(0, 1)
    if norm_upper is None:
        v = data_upper[np.isfinite(data_upper)]
        norm_upper = Normalize(vmin=v.min(), vmax=v.max()) if v.size else Normalize(0, 1)

    cm_lower = mpl.cm.get_cmap(cmap_lower)
    cm_upper = mpl.cm.get_cmap(cmap_upper)

    fig, ax = plt.subplots(figsize=figsize, dpi=350, constrained_layout=False)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    for i in range(nrows):
        for j in range(ncols):
            x0, x1 = j, j + 1
            y0, y1 = i, i + 1

            tri_lower = np.array([[x1, y0], [x1, y1], [x0, y0]])
            tri_upper = np.array([[x0, y1], [x0, y0], [x1, y1]])

            v_lo = data_lower[i, j]
            v_up = data_upper[i, j]

            face_lo = cm_lower(norm_lower(v_lo)) if np.isfinite(v_lo) else (0.9, 0.9, 0.9, 1)
            face_up = cm_upper(norm_upper(v_up)) if np.isfinite(v_up) else (0.9, 0.9, 0.9, 1)

            ax.add_patch(Polygon(tri_lower, closed=True,
                                 facecolor=face_lo, edgecolor="black", linewidth=0.5))
            ax.add_patch(Polygon(tri_upper, closed=True,
                                 facecolor=face_up, edgecolor="black", linewidth=0.5))

            if annotate:
                if np.isfinite(v_lo):
                    ax.text(j + 0.28, i + 0.72, _fmtval(fmt, v_lo),
                            ha="center", va="center", fontsize=textsize,
                            color=_text_color(face_lo))
                if np.isfinite(v_up):
                    ax.text(j + 0.65, i + 0.28, _fmtval(fmt, v_up),
                            ha="center", va="center", fontsize=textsize,
                            color=_text_color(face_up))

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    ax.set_xticks(np.arange(ncols) + 0.5)
    ax.set_yticks(np.arange(nrows) + 0.5)
    ax.set_xticklabels(col_labels, rotation=45, ha="right",
                       rotation_mode="anchor", fontsize=textsize + addsize)
    if show_row_labels:
        ax.set_yticklabels(row_labels, fontsize=textsize + addsize)
    else:
        ax.set_yticklabels([])

    ax.tick_params(axis="y", length=4, width=1)
    ax.tick_params(axis="x", length=4, width=1)
    for spine in ax.spines.values():
        spine.set_visible(True)

    if title:
        ax.set_title(title, pad=4, fontsize=textsize + 2 + addsize)

    if cbar:
        fig.canvas.draw()
        bbox = ax.get_position()

        sm_lo = mpl.cm.ScalarMappable(norm=norm_lower, cmap=cm_lower)
        sm_up = mpl.cm.ScalarMappable(norm=norm_upper, cmap=cm_upper)
        sm_lo.set_array([])
        sm_up.set_array([])

        cax1 = fig.add_axes([bbox.x1 + gap,
                             bbox.y0, cbar_width, bbox.height])
        cax2 = fig.add_axes([bbox.x1 + gap + cbar_width + gap_between_cbars,
                             bbox.y0, cbar_width, bbox.height])

        cb1 = fig.colorbar(sm_lo, cax=cax1)
        cb1.set_label(cbar_label_lower, fontsize=textsize + 2, labelpad=0)
        cb1.ax.tick_params(labelsize=textsize + 2 + addsize)

        cb2 = fig.colorbar(sm_up, cax=cax2)
        cb2.set_label(cbar_label_upper, fontsize=textsize + 2 + addsize, labelpad=1)
        cb2.ax.tick_params(labelsize=textsize + 2 + addsize)

    else:
        plt.tight_layout()

    fig.canvas.draw()

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=350, pad_inches=0.05)

    plt.show()
    plt.close(fig)

    return fig, ax


# ═════════════════════════════════════════════════════════════════════════════
# Utility
# ═════════════════════════════════════════════════════════════════════════════

def merge_mae_rmse_as_string(mae_df, rmse_df, fmt="{:.3f}"):
    """
    Combine two numeric DataFrames into a string DataFrame with cells
    formatted as "mae / rmse".

    Parameters
    ----------
    mae_df, rmse_df : pd.DataFrame – must share index and columns
    fmt             : str – Python format string applied to each value

    Returns
    -------
    pd.DataFrame of strings, same shape as the inputs
    """
    result = pd.DataFrame("", index=mae_df.index, columns=mae_df.columns)
    for col in mae_df.columns:
        for idx in mae_df.index:
            try:
                m = float(mae_df.loc[idx, col])
                r = float(rmse_df.loc[idx, col])
                result.loc[idx, col] = (
                    f"{fmt.format(m)} / {fmt.format(r)}"
                    if np.isfinite(m) and np.isfinite(r) else "—"
                )
            except (TypeError, ValueError):
                result.loc[idx, col] = "—"
    return result


def single_heatmap(
    data,
    row_labels, col_labels,
    title=None,
    cmap="viridis",
    norm=None,
    annotate=True,
    fmt="{:.1f}",
    sig_figs=None,             # significant figures: overrides fmt (e.g. sig_figs=2 → 2 sig figs)
    textsize=8,
    addsize=0,
    cbar=True,
    cbar_label="[%]",
    figsize=(3.5, 5.5),
    gap=0.025,
    cbar_width=0.015,
    left=0.22, right=0.80, bottom=0.02, top=1.0,
    savepath=None,
    show_row_labels=False,
):
    """
    Standard rectangular heatmap with one value per cell.

    Layout parameters match the style of split_triangle_heatmap.

    Parameters
    ----------
    data       : 2D array-like, shape (n_models, n_cols)
    row_labels : list of str – model names (top → bottom)
    col_labels : list of str – column header labels
    cmap       : str – Matplotlib colormap name
    norm       : Normalize or None – auto-computed from data if None
    fmt        : format string for cell annotations
    textsize   : int – base font size for annotations
    addsize    : int – added to textsize for tick labels and title
    gap        : float – gap from heatmap right edge to colorbar
    cbar_width : float – colorbar width (figure-fraction units)
    left, right, bottom, top : float – subplots_adjust margins
    savepath   : str or None – save path (e.g. "out.svg"); None = don't save

    Returns
    -------
    fig, ax
    """
    if sig_figs is not None:
        fmt = _sigfig_formatter(sig_figs)
    data = np.asarray(data, float)
    nrows, ncols = data.shape

    if norm is None:
        v = data[np.isfinite(data)]
        norm = Normalize(vmin=v.min(), vmax=v.max()) if v.size else Normalize(0, 1)

    cm = mpl.cm.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=figsize, dpi=350, constrained_layout=False)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    for i in range(nrows):
        for j in range(ncols):
            val  = data[i, j]
            face = cm(norm(val)) if np.isfinite(val) else (0.9, 0.9, 0.9, 1)
            ax.add_patch(Rectangle((j, i), 1, 1,
                                   facecolor=face, edgecolor="black", linewidth=0.5))
            if annotate and np.isfinite(val):
                ax.text(j + 0.5, i + 0.5, _fmtval(fmt, val),
                        ha="center", va="center", fontsize=textsize,
                        color=_text_color(face))

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    ax.set_xticks(np.arange(ncols) + 0.5)
    ax.set_yticks(np.arange(nrows) + 0.5)
    ax.set_xticklabels(col_labels, rotation=45, ha="right",
                       rotation_mode="anchor", fontsize=textsize + addsize)
    if show_row_labels:
        ax.set_yticklabels(row_labels, fontsize=textsize + addsize)
    else:
        ax.set_yticklabels([])

    ax.tick_params(axis="y", length=4, width=1)
    ax.tick_params(axis="x", length=4, width=1)
    for spine in ax.spines.values():
        spine.set_visible(True)

    if title:
        ax.set_title(title, pad=4, fontsize=textsize + 2 + addsize)

    if cbar:
        fig.canvas.draw()
        bbox = ax.get_position()
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
        sm.set_array([])
        cax = fig.add_axes([bbox.x1 + gap, bbox.y0, cbar_width, bbox.height])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label(cbar_label, fontsize=textsize + 2, labelpad=0)
        cb.ax.tick_params(labelsize=textsize + 2 + addsize)

    else:
        plt.tight_layout()

    fig.canvas.draw()

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=350, pad_inches=0.05)

    plt.show()
    plt.close(fig)

    return fig, ax


# ═════════════════════════════════════════════════════════════════════════════
# Merged two-panel figure: single_heatmap (left) + split_triangle (right)
# ═════════════════════════════════════════════════════════════════════════════

def merged_heatmaps(
    # ── Left panel data (single_heatmap) ─────────────────────────────────
    data_left,
    col_labels_left,
    title_left=None,
    cmap_left="viridis",
    norm_left=None,
    fmt_left="{:.1f}",
    sig_figs_left=None,        # significant figures for left panel (e.g. sig_figs_left=2)
    textsize_left=8,           # cell-number size in the left panel only
    cbar_label_left="[%]",
    annotate_left=True,

    # ── Right panel data (split_triangle_heatmap) ─────────────────────────
    data_lower=None,
    data_upper=None,
    col_labels_right=None,
    title_right=None,
    cmap_lower="viridis",
    cmap_upper="viridis",
    norm_lower=None,
    norm_upper=None,
    fmt_right="{:.1f}",
    sig_figs_right=None,       # significant figures for right panel (e.g. sig_figs_right=2)
    textsize_right=7.5,        # cell-number size in the right panel only
    cbar_label_lower="[%] (<1°)",
    cbar_label_upper="[%] (<20°)",
    annotate_right=True,

    # ── Shared font size (titles, tick labels, cbar labels, suptitle) ────
    fontsize=9,

    # ── Shared ────────────────────────────────────────────────────────────
    row_labels=None,
    suptitle=None,
    suptitle_y=1.01,
    suptitle_x=0.5,            # 0.5 = figure centre; adjust if panels are off-centre
    show_row_labels_left=True,   # right panel always hides row labels

    # ── Figure layout ─────────────────────────────────────────────────────
    figsize=(7.5, 5.5),
    dpi=350,
    bottom=0.025,
    top=0.92,
    left_margin=0.12,          # left edge of left axes (figure fraction)

    # ── Panel spacing ─────────────────────────────────────────────────────
    gap_between_panels=0.06,   # gap between left colorbar and right axes

    # ── Colorbar tuning ───────────────────────────────────────────────────
    cbar_width=0.015,          # width of every colorbar (figure fraction)
    gap_cbar_left=0.025,       # left panel: axes → colorbar gap
    gap_cbar_right=0.02,       # right panel: axes → first colorbar gap
    gap_between_cbars=0.12,    # right panel: cbar1 → cbar2 gap

    savepath=None,
):
    """
    Combine single_heatmap (left) and split_triangle_heatmap (right) into one
    figure, with full control over spacing, colorbar placement, and font sizes.

    Font sizes
    ----------
    fontsize       : controls everything except cell numbers — titles, tick
                     labels, colorbar labels, suptitle all use this value
    textsize_left  : size of the numbers printed inside left-panel cells
    textsize_right : size of the numbers printed inside right-panel cells

    Other parameters
    ----------------
    data_left              : 2-D array for the left (single) heatmap
    col_labels_left        : column labels for the left panel
    data_lower, data_upper : 2-D arrays for the right (split-triangle) panel
    col_labels_right       : column labels for the right panel
    row_labels             : shared row labels (same order for both panels)
    suptitle               : overall figure title (None = no title)
    show_row_labels_left   : show model names on the left panel y-axis
    left_margin            : figure-fraction x offset to the first axes
    gap_between_panels     : figure-fraction gap between left cbar and right axes
    gap_cbar_left/right    : axes → first colorbar gap for each panel
    gap_between_cbars      : gap between the two colorbars of the right panel
    cbar_width             : width of every colorbar in figure fraction

    Returns
    -------
    fig, (ax_left, ax_right)
    """
    if sig_figs_left  is not None:
        fmt_left  = _sigfig_formatter(sig_figs_left)
    if sig_figs_right is not None:
        fmt_right = _sigfig_formatter(sig_figs_right)
    data_left  = np.asarray(data_left,  float)
    data_lower = np.asarray(data_lower, float)
    data_upper = np.asarray(data_upper, float)
    assert data_lower.shape == data_upper.shape
    assert data_left.shape[0] == data_lower.shape[0], "Both panels must have the same number of rows"

    nrows        = data_left.shape[0]
    ncols_left   = data_left.shape[1]
    ncols_right  = data_lower.shape[1]

    # ── norms ────────────────────────────────────────────────────────────
    def _autonorm(arr, n):
        v = arr[np.isfinite(arr)]
        return n if n is not None else (Normalize(v.min(), v.max()) if v.size else Normalize(0, 1))

    norm_left   = _autonorm(data_left,   norm_left)
    norm_lower  = _autonorm(data_lower,  norm_lower)
    norm_upper  = _autonorm(data_upper,  norm_upper)

    cm_left   = mpl.cm.get_cmap(cmap_left)
    cm_lower  = mpl.cm.get_cmap(cmap_lower)
    cm_upper  = mpl.cm.get_cmap(cmap_upper)

    # ── axes geometry ─────────────────────────────────────────────────────
    # With set_aspect("equal") the rectangle must have width = h * ncols/nrows
    # (in figure-fraction space: scale by figsize[1]/figsize[0]).
    h       = top - bottom
    scale   = figsize[1] / figsize[0]
    w_left  = h * scale * ncols_left  / nrows
    w_right = h * scale * ncols_right / nrows

    x0_left     = left_margin
    x0_cbar_l   = x0_left  + w_left  + gap_cbar_left
    x0_right    = x0_cbar_l + cbar_width + gap_between_panels
    x0_cbar_r1  = x0_right + w_right + gap_cbar_right
    x0_cbar_r2  = x0_cbar_r1 + cbar_width + gap_between_cbars

    # ── create figure and axes ────────────────────────────────────────────
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=False)

    ax_left  = fig.add_axes([x0_left,    bottom, w_left,     h])
    ax_right = fig.add_axes([x0_right,   bottom, w_right,    h])
    cax_l    = fig.add_axes([x0_cbar_l,  bottom, cbar_width, h])
    cax_r1   = fig.add_axes([x0_cbar_r1, bottom, cbar_width, h])
    cax_r2   = fig.add_axes([x0_cbar_r2, bottom, cbar_width, h])

    # ── draw left panel ───────────────────────────────────────────────────
    for i in range(nrows):
        for j in range(ncols_left):
            val  = data_left[i, j]
            face = cm_left(norm_left(val)) if np.isfinite(val) else (0.9, 0.9, 0.9, 1)
            ax_left.add_patch(Rectangle((j, i), 1, 1,
                                        facecolor=face, edgecolor="black", linewidth=0.5))
            if annotate_left and np.isfinite(val):
                ax_left.text(j + 0.5, i + 0.5, _fmtval(fmt_left, val),
                             ha="center", va="center", fontsize=textsize_left,
                             color=_text_color(face))

    ax_left.set_xlim(0, ncols_left)
    ax_left.set_ylim(0, nrows)
    ax_left.invert_yaxis()
    ax_left.set_aspect("equal")
    ax_left.set_xticks(np.arange(ncols_left) + 0.5)
    ax_left.set_yticks(np.arange(nrows) + 0.5)
    ax_left.set_xticklabels(col_labels_left, rotation=45, ha="right",
                            rotation_mode="anchor", fontsize=fontsize)
    if show_row_labels_left and row_labels is not None:
        ax_left.set_yticklabels(row_labels, fontsize=fontsize)
    else:
        ax_left.set_yticklabels([])
    ax_left.tick_params(axis="both", length=4, width=1)
    for sp in ax_left.spines.values():
        sp.set_visible(True)
    if title_left:
        ax_left.set_title(title_left, pad=4, fontsize=fontsize)

    sm_l = mpl.cm.ScalarMappable(norm=norm_left, cmap=cm_left)
    sm_l.set_array([])
    cb_l = fig.colorbar(sm_l, cax=cax_l)
    cb_l.set_label(cbar_label_left, fontsize=fontsize, labelpad=0)
    cb_l.ax.tick_params(labelsize=fontsize)

    # ── draw right panel ──────────────────────────────────────────────────
    for i in range(nrows):
        for j in range(ncols_right):
            x0c, x1c = j, j + 1
            y0c, y1c = i, i + 1
            tri_lo = np.array([[x1c, y0c], [x1c, y1c], [x0c, y0c]])
            tri_up = np.array([[x0c, y1c], [x0c, y0c], [x1c, y1c]])
            v_lo = data_lower[i, j]
            v_up = data_upper[i, j]
            face_lo = cm_lower(norm_lower(v_lo)) if np.isfinite(v_lo) else (0.9, 0.9, 0.9, 1)
            face_up = cm_upper(norm_upper(v_up)) if np.isfinite(v_up) else (0.9, 0.9, 0.9, 1)
            ax_right.add_patch(Polygon(tri_lo, closed=True,
                                       facecolor=face_lo, edgecolor="black", linewidth=0.5))
            ax_right.add_patch(Polygon(tri_up, closed=True,
                                       facecolor=face_up, edgecolor="black", linewidth=0.5))
            if annotate_right:
                if np.isfinite(v_lo):
                    ax_right.text(j + 0.28, i + 0.72, _fmtval(fmt_right, v_lo),
                                  ha="center", va="center", fontsize=textsize_right,
                                  color=_text_color(face_lo))
                if np.isfinite(v_up):
                    ax_right.text(j + 0.65, i + 0.28, _fmtval(fmt_right, v_up),
                                  ha="center", va="center", fontsize=textsize_right,
                                  color=_text_color(face_up))

    ax_right.set_xlim(0, ncols_right)
    ax_right.set_ylim(0, nrows)
    ax_right.invert_yaxis()
    ax_right.set_aspect("equal")
    ax_right.set_xticks(np.arange(ncols_right) + 0.5)
    ax_right.set_yticks(np.arange(nrows) + 0.5)
    ax_right.set_xticklabels(col_labels_right, rotation=45, ha="right",
                             rotation_mode="anchor", fontsize=fontsize)
    ax_right.set_yticklabels([])   # row labels shown on left panel only
    ax_right.tick_params(axis="both", length=4, width=1)
    for sp in ax_right.spines.values():
        sp.set_visible(True)
    if title_right:
        ax_right.set_title(title_right, pad=4, fontsize=fontsize)

    sm_lo = mpl.cm.ScalarMappable(norm=norm_lower, cmap=cm_lower)
    sm_up = mpl.cm.ScalarMappable(norm=norm_upper, cmap=cm_upper)
    sm_lo.set_array([])
    sm_up.set_array([])
    cb_r1 = fig.colorbar(sm_lo, cax=cax_r1)
    cb_r1.set_label(cbar_label_lower, fontsize=fontsize, labelpad=0)
    cb_r1.ax.tick_params(labelsize=fontsize)
    cb_r2 = fig.colorbar(sm_up, cax=cax_r2)
    cb_r2.set_label(cbar_label_upper, fontsize=fontsize, labelpad=1)
    cb_r2.ax.tick_params(labelsize=fontsize)

    # ── overall title ─────────────────────────────────────────────────────
    if suptitle:
        fig.suptitle(suptitle, fontsize=fontsize, x=suptitle_x, y=suptitle_y,
                     ha="center")

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=dpi, pad_inches=0.05)

    plt.show()
    return fig, (ax_left, ax_right)


# ═════════════════════════════════════════════════════════════════════════════
# F_DFT fraction and conditioned MAE/RMSE
# ═════════════════════════════════════════════════════════════════════════════

def build_F_dft_frac_table(all_results, fdft_thresholds):
    """
    Fraction (%) of atoms with |F_DFT| > threshold.

    No fdf_min filtering — the whole atom pool is used as denominator
    so the numbers reflect the actual force-magnitude distribution.

    Parameters
    ----------
    all_results      : dict[model → dict]  key: "F_dft"
    fdft_thresholds  : list of float – |F_DFT| cut-offs (eV/Å)

    Returns
    -------
    pd.DataFrame – rows = models, columns = thresholds, values = fraction (%)
    """
    rows = {}
    for model, data in all_results.items():
        F_dft = np.abs(np.asarray(data.get("F_dft", data.get("all_F_dft_mags")), float))
        valid = np.isfinite(F_dft)
        F_f   = F_dft[valid]
        rows[model] = {
            thr: (round(np.mean(F_f > thr) * 100, 2) if F_f.size > 0 else np.nan)
            for thr in fdft_thresholds
        }
    df = pd.DataFrame(rows).T
    df.columns = fdft_thresholds
    return df


def build_F_dft_conditioned_mae_rmse(all_results, fdft_thresholds):
    """
    MAE and RMSE of |ΔF| for atoms with |F_DFT| > threshold.

    Conditioning on larger DFT forces reveals how prediction error
    scales with force magnitude.

    Parameters
    ----------
    all_results     : dict[model → dict]  keys: "F_dft", "deltaF"
    fdft_thresholds : list of float – |F_DFT| cut-offs (eV/Å)

    Returns
    -------
    (mae_df, rmse_df) : two pd.DataFrames
        rows = models, columns = thresholds, values in eV/Å
    """
    mae_rows, rmse_rows = {}, {}

    for model, data in all_results.items():
        F_dft = np.abs(np.asarray(data.get("F_dft", data.get("all_F_dft_mags")),  float))
        dF    = np.asarray(data.get("deltaF", data.get("all_deltaF")), float)
        n     = min(len(F_dft), len(dF))
        F_dft, dF = F_dft[:n], dF[:n]

        valid = np.isfinite(F_dft) & np.isfinite(dF)
        F_f   = F_dft[valid]
        dF_f  = np.abs(dF[valid])

        mae_rows[model], rmse_rows[model] = {}, {}
        for thr in fdft_thresholds:
            sub = dF_f[F_f > thr]
            if sub.size > 0:
                mae_rows[model][thr]  = round(float(np.mean(sub)), 4)
                rmse_rows[model][thr] = round(float(np.sqrt(np.mean(sub ** 2))), 4)
            else:
                mae_rows[model][thr]  = np.nan
                rmse_rows[model][thr] = np.nan

    mae_df  = pd.DataFrame(mae_rows).T;  mae_df.columns  = fdft_thresholds
    rmse_df = pd.DataFrame(rmse_rows).T; rmse_df.columns = fdft_thresholds
    return mae_df, rmse_df


def build_F_dft_conditioned_theta_mae_rmse(all_results, fdft_thresholds):
    """
    MAE and RMSE of |Δθ| for atoms with |F_DFT| > threshold.

    Mirrors build_F_dft_conditioned_mae_rmse but for angle errors.

    Parameters
    ----------
    all_results     : dict[model → dict]  keys: "F_dft", "deltaTheta"
    fdft_thresholds : list of float – |F_DFT| cut-offs (eV/Å)

    Returns
    -------
    (mae_df, rmse_df) : two pd.DataFrames
        rows = models, columns = thresholds, values in degrees
    """
    mae_rows, rmse_rows = {}, {}

    for model, data in all_results.items():
        F_dft  = np.abs(np.asarray(data.get("F_dft", data.get("all_F_dft_mags")), float))
        dtheta = np.abs(np.asarray(data.get("deltaTheta", data.get("all_deltaTheta")), float))
        n      = min(len(F_dft), len(dtheta))
        F_dft, dtheta = F_dft[:n], dtheta[:n]

        valid  = np.isfinite(F_dft) & np.isfinite(dtheta)
        F_f    = F_dft[valid]
        dth_f  = dtheta[valid]

        mae_rows[model], rmse_rows[model] = {}, {}
        for thr in fdft_thresholds:
            sub = dth_f[F_f > thr]
            if sub.size > 0:
                mae_rows[model][thr]  = round(float(np.mean(sub)), 4)
                rmse_rows[model][thr] = round(float(np.sqrt(np.mean(sub ** 2))), 4)
            else:
                mae_rows[model][thr]  = np.nan
                rmse_rows[model][thr] = np.nan

    mae_df  = pd.DataFrame(mae_rows).T;  mae_df.columns  = fdft_thresholds
    rmse_df = pd.DataFrame(rmse_rows).T; rmse_df.columns = fdft_thresholds
    return mae_df, rmse_df


def build_dF_lt_conditioned_mae_rmse(all_results, dF_thresholds, fdft_min=0.01):
    """
    MAE and RMSE of |ΔF| for atoms with |ΔF| < threshold AND |F_DFT| > fdft_min.

    Shows how well the MLIP captures the "easy" (small-error) regime:
    as the threshold rises, more atoms are included and MAE/RMSE grows.

    Parameters
    ----------
    all_results    : dict[model → dict]  keys: "F_dft"/"all_F_dft_mags", "deltaF"/"all_deltaF"
    dF_thresholds  : list of float – upper |ΔF| bounds (eV/Å)
    fdft_min       : float – minimum |F_DFT| for inclusion (eV/Å)

    Returns
    -------
    (mae_df, rmse_df) : two pd.DataFrames
        rows = models, columns = thresholds, values in eV/Å
    """
    mae_rows, rmse_rows = {}, {}

    for model, data in all_results.items():
        F_dft = np.abs(np.asarray(data.get("F_dft", data.get("all_F_dft_mags")), float))
        dF    = np.abs(np.asarray(data.get("deltaF", data.get("all_deltaF")),     float))
        n     = min(len(F_dft), len(dF))
        F_dft, dF = F_dft[:n], dF[:n]

        valid = np.isfinite(F_dft) & np.isfinite(dF) & (F_dft > fdft_min)
        dF_f  = dF[valid]

        mae_rows[model], rmse_rows[model] = {}, {}
        for thr in dF_thresholds:
            sub = dF_f[dF_f < thr]
            if sub.size > 0:
                mae_rows[model][thr]  = round(float(np.mean(sub)), 4)
                rmse_rows[model][thr] = round(float(np.sqrt(np.mean(sub ** 2))), 4)
            else:
                mae_rows[model][thr]  = np.nan
                rmse_rows[model][thr] = np.nan

    mae_df  = pd.DataFrame(mae_rows).T;  mae_df.columns  = dF_thresholds
    rmse_df = pd.DataFrame(rmse_rows).T; rmse_df.columns = dF_thresholds
    return mae_df, rmse_df


# ═════════════════════════════════════════════════════════════════════════════
# Δθ fraction and conditioned MAE/RMSE
# ═════════════════════════════════════════════════════════════════════════════

def build_theta_frac_table(all_results, theta_thresholds, fdf_min=0.01):
    """
    Fraction (%) of atoms with |Δθ| < threshold.

    Only atoms with |F_DFT| > fdf_min are included (angular errors are
    ill-conditioned for near-zero DFT forces).

    Parameters
    ----------
    all_results       : dict[model → dict]  keys: "F_dft", "deltaTheta"
    theta_thresholds  : list of float – angular cut-offs (degrees)
    fdf_min           : float – minimum |F_DFT| for inclusion (eV/Å)

    Returns
    -------
    pd.DataFrame – rows = models, columns = thresholds, values = fraction (%)
    """
    rows = {}
    for model, data in all_results.items():
        F_dft  = np.abs(np.asarray(data.get("F_dft", data.get("all_F_dft_mags")),      float))
        dtheta = np.abs(np.asarray(data.get("deltaTheta", data.get("all_deltaTheta")), float))
        n      = min(len(F_dft), len(dtheta))
        F_dft, dtheta = F_dft[:n], dtheta[:n]

        valid  = np.isfinite(F_dft) & np.isfinite(dtheta) & (F_dft > fdf_min)
        dth_f  = dtheta[valid]

        rows[model] = {
            thr: (round(np.mean(dth_f < thr) * 100, 2) if dth_f.size > 0 else np.nan)
            for thr in theta_thresholds
        }
    df = pd.DataFrame(rows).T
    df.columns = theta_thresholds
    return df


def build_theta_conditioned_mae_rmse(all_results, theta_thresholds, fdf_min=0.01):
    """
    MAE and RMSE of |Δθ| for atoms with |Δθ| < threshold.

    As the threshold increases, more atoms (those with larger angular errors)
    are included, so MAE/RMSE grows monotonically.

    Parameters
    ----------
    all_results      : dict[model → dict]  keys: "F_dft", "deltaTheta"
    theta_thresholds : list of float – upper angular bounds (degrees)
    fdf_min          : float – minimum |F_DFT| for inclusion (eV/Å)

    Returns
    -------
    (mae_df, rmse_df) : two pd.DataFrames
        rows = models, columns = thresholds, values in degrees
    """
    mae_rows, rmse_rows = {}, {}

    for model, data in all_results.items():
        F_dft  = np.abs(np.asarray(data.get("F_dft", data.get("all_F_dft_mags")),      float))
        dtheta = np.abs(np.asarray(data.get("deltaTheta", data.get("all_deltaTheta")), float))
        n      = min(len(F_dft), len(dtheta))
        F_dft, dtheta = F_dft[:n], dtheta[:n]

        valid  = np.isfinite(F_dft) & np.isfinite(dtheta) & (F_dft > fdf_min)
        dth_f  = dtheta[valid]

        mae_rows[model], rmse_rows[model] = {}, {}
        for thr in theta_thresholds:
            sub = dth_f[dth_f < thr]
            if sub.size > 0:
                mae_rows[model][thr]  = round(float(np.mean(sub)), 4)
                rmse_rows[model][thr] = round(float(np.sqrt(np.mean(sub ** 2))), 4)
            else:
                mae_rows[model][thr]  = np.nan
                rmse_rows[model][thr] = np.nan

    mae_df  = pd.DataFrame(mae_rows).T;  mae_df.columns  = theta_thresholds
    rmse_df = pd.DataFrame(rmse_rows).T; rmse_df.columns = theta_thresholds
    return mae_df, rmse_df


def build_theta_regime_panels(all_results, theta_thresh, threshold=1.0, fdf_min=0.01):
    """
    Split atoms into two |F_DFT| regimes and compute Δθ fraction tables.

    Panel A — near-equilibrium    : |F_DFT| < threshold
    Panel B — far-from-equilibrium: |F_DFT| ≥ threshold

    Each row contains:
      "N atoms"               — atom count in this regime
      "Frac of all atoms [%]" — share of the total valid pool
      "|Δθ| < {thr} [%]"     — fraction passing each angular threshold

    Parameters
    ----------
    all_results  : dict[model → dict]  keys: "F_dft", "deltaTheta"
    theta_thresh : list of float – angular thresholds (degrees)
    threshold    : float – |F_DFT| boundary between panels (eV/Å)
    fdf_min      : float – minimum |F_DFT| for inclusion (eV/Å)

    Returns
    -------
    (df_panel_A, df_panel_B) : two pd.DataFrames with rows = models
    """
    panel_A, panel_B = {}, {}

    for model, data in all_results.items():
        F_dft  = np.abs(np.asarray(data.get("F_dft", data.get("all_F_dft_mags")),      float))
        dtheta = np.abs(np.asarray(data.get("deltaTheta", data.get("all_deltaTheta")), float))
        n      = min(len(F_dft), len(dtheta))
        F_dft, dtheta = F_dft[:n], dtheta[:n]

        valid = np.isfinite(F_dft) & np.isfinite(dtheta)
        if fdf_min > 0:
            valid &= F_dft > fdf_min

        F_abs  = F_dft[valid]
        dth    = dtheta[valid]
        total  = F_abs.size

        for panel_dict, mask in [
            (panel_A, F_abs <  threshold),
            (panel_B, F_abs >= threshold),
        ]:
            n_regime = int(mask.sum())
            row = {
                "N atoms":               n_regime,
                "Frac of all atoms [%]": round(100 * n_regime / total, 4) if total > 0 else np.nan,
            }
            for thr in theta_thresh:
                row[f"|Δθ| < {thr} [%]"] = (
                    round(100 * np.mean(dth[mask] < thr), 4) if n_regime > 0 else np.nan
                )
            panel_dict[model] = row

    return pd.DataFrame(panel_A).T, pd.DataFrame(panel_B).T


# ═════════════════════════════════════════════════════════════════════════════
# Additional plot functions
# ═════════════════════════════════════════════════════════════════════════════

def heatmap_fraction_delta_theta(
    df_frac,
    title=None,
    cmap="viridis",
    fmt="{:.1f}",
    sig_figs=None,             # significant figures: overrides fmt (e.g. sig_figs=2 → 2 sig figs)
    textsize=8,
    figsize=(4.5, 5.5),
    cbar_width=0.025,   # width of the vertical colorbar (figure-fraction units)
    cbar_pad=0.01,      # gap between the table right edge and the colorbar
    col_labels=None,    # override auto-generated column labels; None → use "< X°" format
    savepath=None,
):
    """
    imshow-based heatmap for Δθ fraction data.

    Rows = models, columns = angular thresholds (degrees).
    Cell colour encodes the fraction (%) of atoms with |Δθ| < threshold.

    Parameters
    ----------
    df_frac   : pd.DataFrame – output of build_theta_frac_table
    title     : str or None
    cmap      : str – Matplotlib colormap name
    fmt       : str – annotation format string
    textsize  : int – base font size
    figsize   : (w, h) in inches
    savepath  : str or None

    Returns
    -------
    fig, ax
    """
    if sig_figs is not None:
        fmt = _sigfig_formatter(sig_figs)
    data       = df_frac.values.astype(float)
    row_labels = df_frac.index.tolist()
    nrows, ncols = data.shape

    v    = data[np.isfinite(data)]
    norm = Normalize(vmin=v.min(), vmax=v.max()) if v.size else Normalize(0, 100)
    cm   = mpl.cm.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=figsize, dpi=350)
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="equal")

    for i in range(nrows):
        for j in range(ncols):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, _fmtval(fmt, val),
                        ha="center", va="center", fontsize=textsize,
                        color=_text_color(cm(norm(val))))

    ax.set_xticks(np.arange(ncols))
    _col_labels = col_labels if col_labels is not None else [f"< {c}°" for c in df_frac.columns]
    ax.set_xticklabels(
        _col_labels,
        rotation=45, ha="right", rotation_mode="anchor", fontsize=textsize + 1,
    )
    ax.set_yticks(np.arange(nrows))
    ax.set_yticklabels(row_labels, fontsize=textsize + 1)

    if title:
        ax.set_title(title, fontsize=textsize + 2, pad=4)

    # Let tight_layout settle the axes position first (aspect="equal" shrinks it),
    # then place the colorbar manually at the exact axes height.
    plt.tight_layout()
    fig.canvas.draw()
    ax_pos = ax.get_position()

    cax = fig.add_axes([ax_pos.x1 + cbar_pad, ax_pos.y0, cbar_width, ax_pos.height])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("[%]", fontsize=textsize + 1)
    cb.ax.tick_params(labelsize=textsize)

    fig.canvas.draw()

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=350, pad_inches=0.05)

    plt.show()
    plt.close(fig)
    return fig, ax


def triangular_heatmap_with_fraction_row_word_style(
    mae_df, rmse_df, frac_row_str,
    col_labels=None,
    title=r"(MAE / RMSE) $\Delta F$ [eV/Å]",
    cmap_mae="Blues",
    cmap_rmse="Reds",
    figsize=(6.3, 8),
    fmt_mae="{:.2f}",
    fmt_rmse="{:.2f}",
    sig_figs=None,             # significant figures: sets both fmt_mae and fmt_rmse (e.g. sig_figs=2)
    text_size=12,
    cbar_height=0.025,       # thickness of each colorbar (figure-fraction units)
    cbar_gap=0.045,          # gap between table bottom and first (MAE) colorbar
    cbar_between_gap=0.045,  # gap between MAE colorbar and RMSE colorbar
    bottom=0.32,             # figure bottom margin (must fit both bars + gaps)
    savepath=None,
):
    """
    Triangular heatmap with a fraction header row and **horizontal colorbars
    placed below** the axes (the "word-style" variant of the standard function).

    Layout
    ------
    Top row   : pre-formatted fraction strings (white cells, word-style labels)
    Data rows : diagonal-split MAE / RMSE cells
    Below ax  : MAE colorbar, then RMSE colorbar, each spanning the full table width

    Spacing controls
    ----------------
    cbar_height     : thickness of each colorbar bar (figure-fraction units)
    cbar_gap        : vertical gap between the table bottom and the MAE bar
    cbar_between_gap: vertical gap between the MAE bar and the RMSE bar
    bottom          : figure bottom margin reserved for the colorbars
                      (increase if bars are clipped; rule of thumb:
                       bottom ≥ cbar_gap + cbar_height + cbar_between_gap + cbar_height + 0.05)

    Parameters
    ----------
    mae_df, rmse_df : pd.DataFrame – rows = models, columns = thresholds
    frac_row_str    : pd.Series    – pre-formatted strings indexed by threshold
    col_labels      : list or None – override column header labels
    title           : str
    cmap_mae, cmap_rmse : str – Matplotlib colormap names
    figsize         : (w, h) in inches
    fmt_mae, fmt_rmse : str – format strings for cell annotations
    text_size       : int  – base font size
    savepath        : str or None

    Returns
    -------
    fig, ax
    """
    if sig_figs is not None:
        fmt_mae = fmt_rmse = _sigfig_formatter(sig_figs)
    from heatmap_table import (
        draw_rectangular_row, draw_triangular_column,
        setup_frame, setup_ticks_and_labels,
    )

    mae_df, rmse_df = mae_df.align(rmse_df, join="inner", axis=1)
    frac_row_str    = frac_row_str.reindex(mae_df.columns)

    nrows, ncols  = mae_df.shape
    nrows_total   = nrows + 1

    mae_vals  = mae_df.to_numpy(float)
    rmse_vals = rmse_df.to_numpy(float)

    if col_labels is None:
        col_labels = list(mae_df.columns)

    mae_vmin,  mae_vmax  = np.nanpercentile(mae_vals,  [ 5, 95])
    rmse_vmin, rmse_vmax = np.nanpercentile(rmse_vals, [15, 95])

    mae_norm  = plt.Normalize(mae_vmin,  mae_vmax)
    rmse_norm = plt.Normalize(rmse_vmin, rmse_vmax)
    mae_cmap  = mpl.cm.get_cmap(cmap_mae)
    rmse_cmap = mpl.cm.get_cmap(cmap_rmse)

    fig, ax = plt.subplots(figsize=figsize, dpi=350)
    plt.subplots_adjust(bottom=bottom)

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows_total)
    ax.set_aspect("equal")

    # ── fraction header row ────────────────────────────────────────────────
    draw_rectangular_row(ax, row_y=nrows, ncols=ncols,
                         vals=frac_row_str.values,
                         cmap=None, norm=None, text_size=text_size)
    ax.plot([0, ncols], [nrows, nrows], color="black", linewidth=1.0)

    # ── triangular data cells ──────────────────────────────────────────────
    for col_idx in range(ncols):
        draw_triangular_column(
            ax,
            col_idx=col_idx, nrows=nrows,
            vals_lower=mae_vals[:, col_idx],
            vals_upper=rmse_vals[:, col_idx],
            cmap_lower=mae_cmap,  norm_lower=mae_norm,
            cmap_upper=rmse_cmap, norm_upper=rmse_norm,
            row_offset=0,
            fmt_lower=fmt_mae, fmt_upper=fmt_rmse,
            text_size=text_size,
        )

    setup_frame(ax, ncols, nrows_total)
    setup_ticks_and_labels(
        ax,
        ncols=ncols, nrows_data=nrows, nrows_total=nrows_total,
        row_labels=mae_df.index, col_labels=col_labels,
        title=title, text_size=text_size,
    )

    # ── horizontal colorbars below — each spans the full table width ──────────
    fig.canvas.draw()
    ax_pos  = ax.get_position()
    cbar_w  = ax_pos.width
    cbar_x0 = ax_pos.x0

    # MAE bar just below the table, RMSE bar below that
    cbar_y_mae  = ax_pos.y0 - cbar_gap - cbar_height
    cbar_y_rmse = cbar_y_mae - cbar_between_gap - cbar_height

    for cbar_y, cmap_k, norm_k, label_k in [
        (cbar_y_mae,  mae_cmap,  mae_norm,  "MAE [eV/Å]"),
        (cbar_y_rmse, rmse_cmap, rmse_norm, "RMSE [eV/Å]"),
    ]:
        cax = fig.add_axes([cbar_x0, cbar_y, cbar_w, cbar_height])
        sm  = mpl.cm.ScalarMappable(norm=norm_k, cmap=cmap_k)
        sm.set_array([])
        cb  = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label(label_k, fontsize=text_size, labelpad=2)
        cb.ax.tick_params(labelsize=text_size - 1)

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=350, pad_inches=0.05)

    plt.show()
    plt.close(fig)
    return fig, ax


def plot_error_histograms(
    all_results_filtered,
    fdf_min=0.01,
    fdft_max=None,
    bins_dF=100,
    bins_theta=90,
    bins_fdft=100,
    drawstyle="line",
    figsize=(5, 4),
    textsize=10,
    model_colors=None,      # dict {model: color} or None → use default prop_cycle
    model_linestyles=None,  # dict {model: ls} or None → solid "-"
    model_linewidths=None,  # dict {model: lw} or None → use hardcoded default
    savepath_dF=None,
    savepath_theta=None,
    savepath_fdft=None,
):
    """
    Three histograms: |ΔF| (log-x, log-y), |Δθ| (linear-x, log-y),
    and |F_DFT| (log-x, log-y) for all models.

    Parameters
    ----------
    all_results_filtered : dict[model → dict]  keys: "F_dft", "deltaF", "deltaTheta"
    fdf_min        : float – minimum |F_DFT| for inclusion (eV/Å)
    fdft_max       : float or None – maximum |F_DFT| for inclusion; set e.g. 1.0 to
                     restrict all three plots to near-equilibrium atoms only
    bins_dF        : int   – bins for |ΔF| (log-spaced)
    bins_theta     : int   – bins for |Δθ| (linear-spaced 0–180°)
    bins_fdft      : int   – bins for |F_DFT| (log-spaced)
    drawstyle      : str   – "line" (smooth curve) or "step" (step histogram)
    figsize        : (w, h) in inches
    textsize       : int   – base font size
    savepath_dF    : str or None
    savepath_theta : str or None
    savepath_fdft  : str or None
    """
    if model_colors is not None:
        # explicit dict {model: color} — look up per model below
        _color_lookup = model_colors
        colors = None
    else:
        # use the same default prop_cycle that CDF plots use
        _cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        _color_lookup = {m: _cycle[i % len(_cycle)]
                         for i, m in enumerate(all_results_filtered)}
        colors = None

    def _draw(ax, vals, edges, color, label, xlog=False, ls="-", lw=None):
        """Plot one model's histogram in the requested drawstyle."""
        if drawstyle == "step":
            ax.hist(vals, bins=edges, histtype="step",
                    color=color, label=label, lw=(lw or 1.6), linestyle=ls, alpha=0.9)
        else:
            counts, _ = np.histogram(vals, bins=edges)
            centers   = (
                np.sqrt(edges[:-1] * edges[1:]) if xlog
                else 0.5 * (edges[:-1] + edges[1:])
            )
            m = counts > 0
            ax.plot(centers[m], counts[m], color=color, label=label,
                    lw=(lw or 1.5), linestyle=ls)

    fdft_label = (
        rf"  ($|F_{{\mathrm{{DFT}}}}| < {fdft_max}$ eV/Å)" if fdft_max else ""
    )

    # ── |ΔF| histogram (log-x, log-y) ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, dpi=350)
    edges_dF = np.logspace(np.log10(1e-4), np.log10(50), bins_dF + 1)

    for model, data in all_results_filtered.items():
        color = _color_lookup.get(model)
        _ls = (model_linestyles.get(model, "-") if model_linestyles else "-")
        _lw = (model_linewidths.get(model) if model_linewidths else None)
        F_dft = np.abs(np.asarray(data.get("F_dft", data.get("all_F_dft_mags")),  float))
        dF    = np.abs(np.asarray(data.get("deltaF", data.get("all_deltaF")), float))
        n     = min(len(F_dft), len(dF))
        F_dft, dF = F_dft[:n], dF[:n]
        valid = np.isfinite(F_dft) & np.isfinite(dF) & (F_dft > fdf_min) & (dF > 0)
        if fdft_max is not None:
            valid &= F_dft < fdft_max
        _draw(ax, dF[valid], edges_dF, color, model, xlog=True, ls=_ls, lw=_lw)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|\Delta F|$ (eV/Å)", fontsize=textsize)
    ax.set_ylabel("Number of Atoms", fontsize=textsize)
    ax.set_title(r"Histogram of $|\Delta F|$" + fdft_label, fontsize=textsize + 1)
    ax.tick_params(labelsize=textsize - 1)
    ax.legend(fontsize=textsize - 2, loc="best")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    if savepath_dF is not None:
        fig.savefig(savepath_dF, bbox_inches="tight", dpi=350, pad_inches=0.05)
    plt.show()
    plt.close(fig)

    # ── |Δθ| histogram (linear-x, log-y) ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, dpi=350)
    edges_th = np.linspace(0, 180, bins_theta + 1)

    for model, data in all_results_filtered.items():
        color = _color_lookup.get(model)
        _ls = (model_linestyles.get(model, "-") if model_linestyles else "-")
        _lw = (model_linewidths.get(model) if model_linewidths else None)
        F_dft  = np.abs(np.asarray(data.get("F_dft", data.get("all_F_dft_mags")),      float))
        dtheta = np.abs(np.asarray(data.get("deltaTheta", data.get("all_deltaTheta")), float))
        n      = min(len(F_dft), len(dtheta))
        F_dft, dtheta = F_dft[:n], dtheta[:n]
        valid  = np.isfinite(F_dft) & np.isfinite(dtheta) & (F_dft > fdf_min)
        if fdft_max is not None:
            valid &= F_dft < fdft_max
        _draw(ax, dtheta[valid], edges_th, color, model, xlog=False, ls=_ls, lw=_lw)

    ax.set_yscale("log")
    ax.set_xlabel(r"$|\Delta\theta|$ (deg)", fontsize=textsize)
    ax.set_ylabel("Number of Atoms", fontsize=textsize)
    ax.set_title(r"Histogram of $|\Delta\theta|$" + fdft_label, fontsize=textsize + 1)
    ax.tick_params(labelsize=textsize - 1)
    ax.legend(fontsize=textsize - 2, loc="best")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    if savepath_theta is not None:
        fig.savefig(savepath_theta, bbox_inches="tight", dpi=350, pad_inches=0.05)
    plt.show()
    plt.close(fig)

    # ── |F_DFT| histogram (log-x, log-y) ─────────────────────────────────────
    # F_DFT is DFT reference data — identical across models, so plot only once.
    fig, ax = plt.subplots(figsize=figsize, dpi=350)

    first_data = next(iter(all_results_filtered.values()))
    F_dft = np.abs(np.asarray(first_data.get("F_dft", first_data.get("all_F_dft_mags")), float))
    valid = np.isfinite(F_dft) & (F_dft > fdf_min)
    if fdft_max is not None:
        valid &= F_dft < fdft_max

    if fdft_max is not None:
        fdft_upper = fdft_max
    else:
        fdft_upper = float(F_dft[valid].max()) if valid.any() else 100.0
    edges_fd   = np.logspace(np.log10(max(fdf_min, 1e-4)), np.log10(fdft_upper), bins_fdft + 1)
    _draw(ax, F_dft[valid], edges_fd, "steelblue", r"$|F_{\mathrm{DFT}}|$", xlog=True)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|F_{\mathrm{DFT}}|$ (eV/Å)", fontsize=textsize)
    ax.set_ylabel("Number of Atoms", fontsize=textsize)
    ax.set_title(r"Histogram of $|F_{\mathrm{DFT}}|$" + fdft_label, fontsize=textsize + 1)
    ax.tick_params(labelsize=textsize - 1)
    ax.legend(fontsize=textsize - 2, loc="best")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    if savepath_fdft is not None:
        fig.savefig(savepath_fdft, bbox_inches="tight", dpi=350, pad_inches=0.05)
    plt.show()
    plt.close(fig)


def get_bad_atom_indices(all_results, model,
                         dF_gt=None, dF_lt=None,
                         dtheta_gt=None, dtheta_lt=None,
                         fdft_gt=None, fdft_lt=None):
    """
    Return flat atom-level indices and the original structure indices of
    structures that contain at least one matching atom.

    Uses forces_mlip (per-structure) to reconstruct which atoms belong to
    which structure, then maps back via original_indices — no results_*.json needed.

    Parameters
    ----------
    all_results : dict[model → dict]
    model       : str – key into all_results
    dF_gt/lt    : float or None – |ΔF| > / < threshold (eV/Å)
    dtheta_gt/lt: float or None – |Δθ| > / < threshold (degrees)
    fdft_gt/lt  : float or None – |F_DFT| > / < threshold (eV/Å)

    Returns
    -------
    atom_indices      : np.ndarray of int – positions in the flat per-atom arrays
    structure_indices : list of int – original_index of structures with ≥1 matching atom
    """
    d    = all_results[model]
    dF   = np.abs(np.array(d.get("all_deltaF",     d.get("deltaF",     [])), dtype=float))
    dth  = np.abs(np.array(d.get("all_deltaTheta", d.get("deltaTheta", [])), dtype=float))
    fdft = np.abs(np.array(d.get("all_F_dft_mags", d.get("F_dft",      [])), dtype=float))

    mask = np.ones(len(dF), dtype=bool)
    if dF_gt     is not None: mask &= dF   > dF_gt
    if dF_lt     is not None: mask &= dF   < dF_lt
    if dtheta_gt is not None: mask &= dth  > dtheta_gt
    if dtheta_lt is not None: mask &= dth  < dtheta_lt
    if fdft_gt   is not None: mask &= fdft > fdft_gt
    if fdft_lt   is not None: mask &= fdft < fdft_lt

    atom_indices = np.where(mask)[0]

    # Map flat atom indices → structure original_indices via forces_mlip atom counts
    orig_idxs      = d.get("original_indices", [])
    forces_mlip    = d.get("forces_mlip", [])
    structure_indices = []
    if orig_idxs and forces_mlip:
        atom_counts  = np.array([len(f) for f in forces_mlip])
        struct_start = np.concatenate([[0], np.cumsum(atom_counts[:-1])])
        struct_end   = np.cumsum(atom_counts)
        bad_set      = set(atom_indices.tolist())
        for i, (s, e) in enumerate(zip(struct_start, struct_end)):
            if any(a in bad_set for a in range(s, e)):
                structure_indices.append(orig_idxs[i])

    return atom_indices, structure_indices


def get_bad_structure_indices(results_list,
                               dF_gt=None, dF_lt=None,
                               dtheta_gt=None, dtheta_lt=None,
                               fdft_gt=None, fdft_lt=None,
                               require_any=True):
    """
    From a results_*.json list (per-structure records), return the
    original_index of structures where at least one atom (require_any=True)
    or ALL atoms (require_any=False) satisfy the conditions.

    Parameters
    ----------
    results_list : list of dict – loaded from results_*.json
    require_any  : bool – True  → flag structure if ANY atom matches
                          False → flag structure only if ALL atoms match

    Returns
    -------
    list of int – original_index values of matching structures
    """
    bad = []
    for record in results_list:
        dF   = np.abs(np.array(record.get("deltaF",     []), dtype=float))
        dth  = np.abs(np.array(record.get("deltaTheta", []), dtype=float))
        fdft_raw = record.get("forces_dft", None)
        fdft = (np.linalg.norm(np.array(fdft_raw, dtype=float), axis=1)
                if fdft_raw is not None else np.full(len(dF), np.nan))

        n = min(len(dF), len(dth), len(fdft))
        dF, dth, fdft = dF[:n], dth[:n], fdft[:n]

        mask = np.ones(n, dtype=bool)
        if dF_gt     is not None: mask &= dF   > dF_gt
        if dF_lt     is not None: mask &= dF   < dF_lt
        if dtheta_gt is not None: mask &= dth  > dtheta_gt
        if dtheta_lt is not None: mask &= dth  < dtheta_lt
        if fdft_gt   is not None: mask &= fdft > fdft_gt
        if fdft_lt   is not None: mask &= fdft < fdft_lt

        hit = mask.any() if require_any else mask.all()
        if hit:
            bad.append(record["original_index"])
    return bad
