"""
Helpers for notebooks/01_eda.ipynb (US 2.1): plotting style, imbalance statistics and
inter-rater agreement metrics. Kept out of the notebook so the notebook reads as analysis
and the same functions can be reused (e.g. for split audits in US 2.2).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.paths import get_data_root  # noqa: F401  (re-exported for notebook convenience)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "results" / "figures" / "eda"
TABLE_DIR = REPO_ROOT / "results" / "tables" / "eda"

# --- Colour: validated categorical order (blue, orange, aqua, yellow, magenta, green, violet, red) --- #
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# Colour follows the entity, never its rank: fixed mapping so a filter never repaints a series.
L1_ORDER = ["Arabic", "Hindi", "Korean", "Mandarin", "Spanish", "Vietnamese"]
L1_COLOR = dict(zip(L1_ORDER, CAT))
SPLIT_COLOR = {"train": CAT[0], "test": CAT[1], "val": CAT[2]}
ERROR_ORDER = ["substitution", "deletion", "addition"]
ERROR_COLOR = dict(zip(ERROR_ORDER, CAT[:3]))

VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW", "AX"}


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 2, "font.size": 9,
    })


def save_fig(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path


def save_table(df: pd.DataFrame, name: str, index: bool = True) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=index)
    return path


# --------------------------------------------------------------------------- #
# Class-imbalance statistics
# --------------------------------------------------------------------------- #

def imbalance_summary(counts: pd.Series) -> dict:
    """Describe how skewed a categorical target is (counts indexed by class)."""
    counts = counts[counts > 0].sort_values(ascending=False)
    total = counts.sum()
    p = counts / total
    entropy = float(-(p * np.log(p)).sum())
    return {
        "n_classes": int(len(counts)), "n": int(total),
        "majority": str(counts.index[0]), "majority_share": float(p.iloc[0]),
        "minority": str(counts.index[-1]), "minority_share": float(p.iloc[-1]),
        "imbalance_ratio": float(counts.iloc[0] / counts.iloc[-1]),
        # 1.0 = perfectly balanced, 0 = single class
        "norm_entropy": entropy / math.log(len(counts)) if len(counts) > 1 else 0.0,
    }


# --------------------------------------------------------------------------- #
# Inter-rater agreement
# --------------------------------------------------------------------------- #

def mean_pairwise_pearson(wide: pd.DataFrame) -> float:
    """wide: items x raters. Mean off-diagonal Pearson correlation."""
    c = wide.corr().to_numpy()
    return float(c[~np.eye(len(c), dtype=bool)].mean())


def leave_one_out_pearson(wide: pd.DataFrame) -> float:
    """Mean correlation of each rater with the mean of the others: a realistic 'human ceiling' for a
    model that predicts the consensus."""
    vals = []
    for col in wide.columns:
        others = wide.drop(columns=col).mean(axis=1)
        vals.append(np.corrcoef(wide[col], others)[0, 1])
    return float(np.mean(vals))


def icc_1_1(wide: pd.DataFrame) -> float:
    """One-way random ICC(1,1). Used because rater identity is not guaranteed to be consistent
    across utterances, so raters are treated as exchangeable."""
    x = wide.to_numpy(dtype=float)
    n, k = x.shape
    row_means = x.mean(axis=1)
    msb = k * ((row_means - x.mean()) ** 2).sum() / (n - 1)
    msw = ((x - row_means[:, None]) ** 2).sum() / (n * (k - 1))
    return float((msb - msw) / (msb + (k - 1) * msw))


def icc_2_1(wide: pd.DataFrame) -> float:
    """Two-way random ICC(2,1), absolute agreement: appropriate when the same raters score every item
    and systematic rater severity differences should count as disagreement."""
    x = wide.to_numpy(dtype=float)
    n, k = x.shape
    grand, row, col = x.mean(), x.mean(axis=1), x.mean(axis=0)
    msr = k * ((row - grand) ** 2).sum() / (n - 1)
    msc = n * ((col - grand) ** 2).sum() / (k - 1)
    mse = ((x - row[:, None] - col[None, :] + grand) ** 2).sum() / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def fleiss_kappa(counts: np.ndarray) -> float:
    """counts: items x categories, each row summing to the (constant) number of raters."""
    counts = np.asarray(counts, dtype=float)
    n = counts.sum(axis=1)[0]
    p_j = counts.sum(axis=0) / counts.sum()
    p_i = ((counts ** 2).sum(axis=1) - n) / (n * (n - 1))
    p_e = (p_j ** 2).sum()
    return float((p_i.mean() - p_e) / (1 - p_e))


def pairwise_exact_agreement(counts: np.ndarray) -> float:
    """Share of rater pairs that gave the same label (counts: items x categories)."""
    counts = np.asarray(counts, dtype=float)
    n = counts.sum(axis=1)[0]
    return float((((counts ** 2).sum(axis=1) - n) / (n * (n - 1))).mean())


# --------------------------------------------------------------------------- #
# Small plotting helpers
# --------------------------------------------------------------------------- #

def label_bars(ax, fmt="{:,.0f}", horizontal=False, pad=3, **kw) -> None:
    """Value labels at bar ends (selective use: only for bar charts with few bars)."""
    for bar in ax.patches:
        v = bar.get_width() if horizontal else bar.get_height()
        if not np.isfinite(v) or v == 0:
            continue
        if horizontal:
            ax.annotate(fmt.format(v), (v, bar.get_y() + bar.get_height() / 2), xytext=(pad, 0),
                        textcoords="offset points", va="center", ha="left", color=INK_2, fontsize=8, **kw)
        else:
            ax.annotate(fmt.format(v), (bar.get_x() + bar.get_width() / 2, v), xytext=(0, pad),
                        textcoords="offset points", va="bottom", ha="center", color=INK_2, fontsize=8, **kw)


def heatmap(ax, data: pd.DataFrame, fmt="{:.0f}", vmax=None, label_cells=True, cbar_label=None):
    """Sequential one-hue heatmap (light = near zero)."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    vmax = vmax if vmax is not None else np.nanmax(data.to_numpy())
    im = ax.imshow(data.to_numpy(dtype=float), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(data.shape[1]), data.columns, rotation=45, ha="right")
    ax.set_yticks(range(data.shape[0]), data.index)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if label_cells:
        arr = data.to_numpy(dtype=float)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if np.isfinite(v):
                    ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5,
                            color="white" if v > 0.55 * vmax else INK)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    if cbar_label:
        cb.set_label(cbar_label, color=INK_2)
    return im
