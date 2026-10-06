"""US 3.2 figures (validation data only) -> results/figures/us32/"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[2]
A, F = R / "results" / "analysis" / "us32", R / "results" / "figures" / "us32"
F.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GRAY = "#2a6fbb", "#d9822b", "#6b7280"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRAY,
                     "axes.labelcolor": "#222", "xtick.color": "#222", "ytick.color": "#222"})

# 1. confusion matrix (row-normalised, with counts)
cm = pd.read_csv(A / "l2_phone_confusion_rownorm_val.csv", index_col=0)
cnt = pd.read_csv(A / "l2_phone_confusion_counts_val.csv", index_col=0)
fig, ax = plt.subplots(figsize=(5.6, 4.4))
im = ax.imshow(cm.values, cmap="Blues", vmin=0, vmax=1)
labs = [c.replace("pred_", "") for c in cm.columns]
ax.set_xticks(range(4), labs); ax.set_yticks(range(4), labs)
ax.set_xlabel("predicted"); ax.set_ylabel("true")
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{cm.values[i, j]:.1%}\n(n={cnt.values[i, j]:,})", ha="center", va="center", fontsize=8,
                color="white" if cm.values[i, j] > 0.5 else "#222")
ax.set_title("L2-ARCTIC phone error type, SVM baseline (validation, row-normalised)", fontsize=9)
fig.tight_layout(); fig.savefig(F / "l2_phone_confusion_val.png", dpi=160); plt.close(fig)

# 2. regression to the mean
b = pd.read_csv(A / "regression_by_score_bin_val.csv")
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, t, title in zip(axes, ["sent_total", "word_accuracy", "phone_score"], ["Sentence total (0-10)", "Word accuracy (0-10)", "Phone score (0-2)"]):
    g = b[b.task == t]
    ax.plot(g.mean_true, g.mean_true, color=GRAY, lw=1.5, ls="--", label="perfect")
    ax.plot(g.mean_true, g.mean_pred, color=BLUE, lw=2, marker="o", ms=6, label="mean prediction")
    ax.set_title(title, fontsize=10); ax.set_xlabel("true score (bin mean)")
axes[0].set_ylabel("predicted"); axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Baseline predictions regress to the mean: low scores are over-predicted, top scores under-predicted (validation)", fontsize=9)
fig.tight_layout(); fig.savefig(F / "regression_to_mean_val.png", dpi=160); plt.close(fig)

# 3. imbalance experiment: recall unweighted vs balanced
e = pd.read_csv(A / "l2_phone_imbalance_experiment_val.csv").set_index("variant")
labels = ["substitution", "deletion", "addition"]
fig, ax = plt.subplots(figsize=(6.2, 3.6))
x = np.arange(len(labels)); w = 0.36
for k, (v, c) in enumerate([("logreg unweighted", BLUE), ("logreg balanced", ORANGE)]):
    vals = [e.loc[v, f"recall_{l}"] for l in labels]
    bars = ax.bar(x + (k - 0.5) * w, vals, w - 0.04, color=c, label=v.replace("logreg ", ""))
    for xi, val in zip(x + (k - 0.5) * w, vals):
        ax.text(xi, val + 0.01, f"{val:.2f}", ha="center", fontsize=8)
ax.set_xticks(x, labels); ax.set_ylabel("recall (validation)"); ax.set_ylim(0, 0.8); ax.legend(frameon=False, fontsize=8)
ax.set_title("Class weighting lifts minority recall (L2-ARCTIC phone error type)", fontsize=9)
fig.tight_layout(); fig.savefig(F / "l2_imbalance_recall_val.png", dpi=160); plt.close(fig)
print("ok")
