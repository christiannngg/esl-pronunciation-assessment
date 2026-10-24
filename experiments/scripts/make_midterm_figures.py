"""Midterm demo figures: the results ladder (baseline -> frozen -> fine-tuned) and the class-imbalance chart.

    python experiments/scripts/make_midterm_figures.py                 # run from the repo root
    python experiments/scripts/make_midterm_figures.py --root . --out results/figures/midterm

Reads the committed headline tables (test set, macro-F1):
  results/tables/finetune/us51_headline.csv  -> columns us31 (best MFCC baseline), us33 (after Sprint 3 fix), frozen_43 (frozen WavLM, US 4.3)
  results/tables/finetune/us53_headline.csv  -> test_after (final fine-tuned system after US 5.3 selection)
Class shares for the imbalance chart are from docs/eda_summary.md (L2-ARCTIC annotated phone events).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reference categorical palette (slots 1-2, validated light mode) and ink tokens.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b9b8b2"

TASKS = {
    "l2_phone_error": ("Phoneme error type (L2-ARCTIC)", BLUE),
    "phone_mispron_t1.0": ("Mispronunciation detection (SpeechOcean762)", ORANGE),
}
STAGES = ["MFCC baseline\n(Sprint 3)", "Baseline +\nweighting fix", "Frozen WavLM\n(Sprint 4)", "Fine-tuned WavLM\n(Sprint 5)"]


def read_rows(path: Path, key: str = "task") -> dict[str, dict]:
    with open(path, newline="") as f:
        return {r[key]: r for r in csv.DictReader(f) if r.get("tag", "base") == "base"}


def ladder_values(root: Path) -> dict[str, list[float]]:
    r51 = read_rows(root / "results/tables/finetune/us51_headline.csv")
    r53 = read_rows(root / "results/tables/finetune/us53_headline.csv")
    out = {}
    for task in TASKS:
        a, b = r51[task], r53[task]
        out[task] = [float(a["us31"]), float(a["us33"]), float(a["frozen_43"]), float(b["test_after"])]
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=15)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def ladder(root: Path, out: Path) -> None:
    vals = ladder_values(root)
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150, facecolor=SURFACE)
    style(ax)
    n = len(TASKS)
    w = 0.36
    for j, (task, (label, color)) in enumerate(TASKS.items()):
        xs = [i + (j - (n - 1) / 2) * (w + 0.02) for i in range(len(STAGES))]
        bars = ax.bar(xs, vals[task], width=w, color=color, label=label, zorder=3)
        for x, v in zip(xs, vals[task]):
            ax.text(x, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=17, fontweight="bold", color=INK)
    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels(STAGES, color=INK, fontsize=16)
    ax.set_ylim(0, 0.85)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_ylabel("Macro-F1 on test set (higher is better)", color=INK2, fontsize=15)
    fig.suptitle("Pretrained speech models beat handcrafted features", x=0.07, ha="left", fontsize=26, fontweight="bold", color=INK, y=0.97)
    fig.text(0.07, 0.895, "Same speakers, same test set at every stage", ha="left", va="top", fontsize=16, color=INK2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=15, labelcolor=INK, ncols=2, columnspacing=2.5)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(out / "results_ladder.png", facecolor=SURFACE)
    plt.close(fig)


def imbalance(out: Path) -> None:
    labels = ["Correct", "Substitution", "Deletion", "Addition"]
    shares = [84.5, 11.7, 2.8, 0.9]
    colors = [GRAY, BLUE, BLUE, BLUE]
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150, facecolor=SURFACE)
    style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.barh(labels[::-1], shares[::-1], color=colors[::-1], height=0.62, zorder=3)
    for y, v in zip(range(len(labels)), shares[::-1]):
        ax.text(v + 1.2, y, f"{v:g}%", va="center", fontsize=20, fontweight="bold", color=INK)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels([f"{t}%" for t in (0, 25, 50, 75, 100)])
    ax.tick_params(axis="y", labelsize=19, labelcolor=INK)
    ax.set_xlabel("Share of annotated phone events (L2-ARCTIC)", color=INK2, fontsize=15)
    fig.suptitle("Most sounds are correct, so errors are rare", x=0.07, ha="left", fontsize=26, fontweight="bold", color=INK, y=0.97)
    ax.set_title("Class imbalance is why the first baseline called almost everything correct", loc="left", fontsize=16, color=INK2, pad=18)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "class_imbalance.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--out", type=Path, default=Path("results/figures/midterm"))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ladder(a.root, a.out)
    imbalance(a.out)
    print("wrote", a.out / "results_ladder.png", "and", a.out / "class_imbalance.png")
