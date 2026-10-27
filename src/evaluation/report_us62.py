"""
US 6.2 - diagnosis of the multi-task models: where does sharing the encoder help or hurt, and why?

    python -m src.evaluation.report_us62                       # transfer + gradient + stopping diagnosis (+ ablations that have been run)
    python -m src.evaluation.report_us62 --no-ablations
    python -m src.evaluation.report_us62 --groups so762

Needs (all written earlier, nothing is trained here):
  * results/tables/multitask/us61_headline_<tag>.csv   (python -m src.evaluation.report_us61 --tag <tag> --groups <group>)
  * experiments/results/multitask/<tag>/_group_<group>/fold*/seed*.npz  (per-epoch gradient diagnostics written by the training jobs)
  * optional: ablation jobs, tags ``mt62_<variant>`` (bash run_62_local.sh ablate <variant>)

What it reports, per group (the recipe of each group is `compare62.group_tags` in configs/multitask.yaml, decided by the US 6.1 validation screen):
  1. Transfer: multi-task minus the single-task model trained with the SAME recipe, test, paired speaker bootstrap (the US 6.1 numbers, matched comparator).
  2. Task imbalance / gradient dominance: each head's share of the summed gradient norms on the last shared transformer layer (1/K = balanced).
  3. Gradient conflict: mean cosine between the heads' gradients on that layer, and the fraction of jobs in which a pair is below the declared threshold.
  4. Stopping compromise: the cost, on validation, of using ONE shared checkpoint instead of each head's own best epoch (an optimistic bound: selected on
     the same validation data), and how far each head has got relative to its head-only epoch (lagging heads).
  5. Ablations (SO762): fewer heads / a changed loss weight; compared with the full multi-task model and with the single-task model.
  6. A cause table: which declared criteria fire for which head. The thresholds are fixed in the config before the diagnostics were looked at.

Everything is descriptive. The gradient measurement is the per-head gradient of the UNWEIGHTED head loss on the trainable parameters of the last transformer
layer (a cheap, standard probe of the shared encoder); cosines and norms are averaged over the sampled steps of every epoch after the head-only epoch.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.evaluation import report_us61 as R61
from src.models import finetune as FT
from src.models import frozen_heads as FH
from src.models import multitask as MT
from src.models.baselines import build_tasks
from src.models.baselines import load_config as load_baseline_config

RU = R61.RU
OUT = FT.REPO / "results" / "tables" / "multitask"
FIG = FT.REPO / "results" / "figures" / "multitask"


# --------------------------------------------------------------------------- #
# pure helpers (numpy only, unit-tested)
# --------------------------------------------------------------------------- #
def job_diag(g) -> dict | None:
    """Per-job gradient diagnostics from a group history file (``hist_gnorm`` (E,K), ``hist_gcos`` (E,K,K), ``hist_gn`` (E,), ``hist_w`` (E,K)).
    Epochs without samples (the head-only epoch) are ignored; epochs are weighted by the number of sampled steps. None when no epoch has samples."""
    gn = np.asarray(g["hist_gn"], float)
    valid = gn > 0
    if not valid.any():
        return None
    norms, cos, w = np.asarray(g["hist_gnorm"], float), np.asarray(g["hist_gcos"], float), np.asarray(g["hist_w"], float)
    wt = gn[valid]
    share_ep = np.full(norms.shape, np.nan)
    for e in np.flatnonzero(valid):
        share_ep[e] = MT.grad_shares(norms[e])
    n = np.average(norms[valid], axis=0, weights=wt)
    return {"norms": n, "shares": np.average(share_ep[valid], axis=0, weights=wt), "cos": np.average(cos[valid], axis=0, weights=wt),
            "eff_shares": MT.grad_shares(np.average(w[valid], axis=0, weights=wt) * n), "shares_epoch": share_ep}


def dominance_label(share: float, k: int, ratio: float) -> str:
    """dominant: share > ratio/K ; starved: share < 1/(ratio*K) ; else balanced."""
    if not np.isfinite(share):
        return "n/a"
    if share > ratio / k:
        return "dominant"
    if share < 1.0 / (ratio * k):
        return "starved"
    return "balanced"


def conflict_pairs(cos_jobs: np.ndarray, names: list[str], thr: float, min_frac: float, aligned: float) -> list[dict]:
    """cos_jobs: (J,K,K) per-job mean cosine matrices. A pair CONFLICTS if its mean cosine < thr in at least ``min_frac`` of the jobs."""
    cos_jobs = np.asarray(cos_jobs, float)
    out = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            v = cos_jobs[:, a, b]
            m = float(np.mean(v))
            frac = float(np.mean(v < thr))
            out.append({"head_a": names[a], "head_b": names[b], "mean_cos": m, "sd_cos": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"),
                        "min_cos": float(v.min()), "frac_jobs_below_thr": frac, "jobs": len(v),
                        "conflict": bool(m < thr and frac >= min_frac), "aligned": bool(m > aligned)})
    return out


def alignment_with_rest(norms: np.ndarray, cos: np.ndarray) -> np.ndarray:
    """Cosine between each head's gradient and the SUM of the other heads' gradients, rebuilt from the (averaged) norms and cosines
    (Gram = n_i n_j cos_ij). An approximation: the average of products is not the product of averages. NaN when K == 1."""
    n, cos = np.asarray(norms, float), np.asarray(cos, float)
    K = len(n)
    G = np.outer(n, n) * cos
    G[np.diag_indices(K)] = n ** 2
    out = np.full(K, np.nan)
    for k in range(K):
        rest = [j for j in range(K) if j != k]
        if not rest:
            continue
        s2 = G[np.ix_(rest, rest)].sum()
        if n[k] > 0 and s2 > 0:
            out[k] = float(G[k, rest].sum() / (n[k] * np.sqrt(s2)))
    return out


def stopping_regret(kind: str, val_primary, shared_epoch: int) -> dict:
    """Validation error at the shared checkpoint vs the head's own best epoch. ``val_primary`` per epoch as FH.primary_score (RMSE or -macro-F1)."""
    err = np.array([MT.error_from_primary(kind, v) for v in np.asarray(val_primary, float)])
    e = int(min(max(shared_epoch, 0), len(err) - 1))
    own = int(np.argmin(err))
    regret = float(err[e] - err[own])
    return {"err_shared": float(err[e]), "err_own": float(err[own]), "regret": regret, "regret_rel": regret / max(float(err[e]), 1e-12), "own_epoch": own}


def matched_comparator(recipe: str, stabilized_variant) -> str:
    """The single-task model trained with the same recipe as the multi-task tag: ``base`` for the base recipe, the US 5.3 model when it IS that recipe."""
    return "stabilized" if recipe != "base" and stabilized_variant == recipe else "base"


def transfer_label(verdict: str) -> str:
    return {"worse": "negative transfer", "better": "positive transfer"}.get(verdict, "no clear transfer")


def ablation_reading(vs_st: str, vs_full: str) -> str:
    """Verdicts (RU.verdict style) of the ablation model against (a) the single-task comparator and (b) the full multi-task model."""
    if vs_full == "better":
        return "recovers" if vs_st != "worse" else "partly"
    return "hurts" if vs_full == "worse" else "no effect"


def pad_stack(arrs: list[np.ndarray]) -> np.ndarray:
    """(J, E_max, K) from per-job (E, K) arrays of different epoch counts, NaN padded."""
    E = max(a.shape[0] for a in arrs)
    out = np.full((len(arrs), E) + arrs[0].shape[1:], np.nan)
    for i, a in enumerate(arrs):
        out[i, :a.shape[0]] = a
    return out


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def collect_group(group: str, tag: str, variant: str | None, registry: dict) -> dict | None:
    """All finished jobs of one (group, tag): gradient diagnostics + per-head validation histories. None when nothing has finished."""
    cfg = MT.load_config(variant)
    names = cfg["groups"][group]["tasks"]
    first = registry[names[0]]
    jobs = []
    for fold in first.folds:
        for seed in MT.seeds_for_group(cfg, registry, group):
            gf = MT.group_file(cfg, tag, group, fold, seed)
            hf = {n: MT.job_file(cfg, tag, n, fold, seed) for n in names}
            if not gf.exists() or not all(f.exists() for f in hf.values()):
                continue
            g = np.load(gf, allow_pickle=True)
            if [str(h) for h in g["heads"]] != list(names):
                raise ValueError(f"{gf}: heads {list(g['heads'])} differ from the config {names}")
            heads = {n: np.load(f, allow_pickle=True) for n, f in hf.items()}
            jobs.append({"fold": fold, "seed": seed, "shared_epoch": int(g["best_epoch"]), "ref_err": np.asarray(g["ref_err"], float),
                         "diag": job_diag(g), "heads": heads, "kinds": [str(k) for k in g["kinds"]]})
    return {"group": group, "tag": tag, "names": list(names), "jobs": jobs} if jobs else None


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def transfer_table(group: str, spec: dict, names: list[str]) -> pd.DataFrame:
    tag = spec["tag"]
    path = OUT / f"us61_headline_{tag}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: python -m src.evaluation.report_us61 --tag {tag} " + (f"--variant {spec['variant']} " if spec.get("variant") else "") + f"--groups {group}")
    df = pd.read_csv(path)
    df = df[df["group"] == group].set_index("task")
    recipe = tag.replace("mt_", "", 1)
    rows = []
    for n in names:
        if n not in df.index:
            continue
        r = df.loc[n]
        comp = matched_comparator(recipe, r.get("stabilized_variant"))
        rows.append({"group": group, "task": n, "metric": r["metric"], "comparator": comp, "st_recipe": r.get(f"st_rule_{comp}", ""), "st": r[f"st_{comp}"], "mt": r["mt"],
                     "delta": r[f"d_{comp}"], "lo": r[f"d_{comp}_lo"], "hi": r[f"d_{comp}_hi"], "p": r[f"d_{comp}_p"], "verdict": r[f"verdict_{comp}"],
                     "transfer": transfer_label(r[f"verdict_{comp}"]), "val_flag": r[f"val_flag_{comp}"], "val_worse_by": r[f"val_worse_by_{comp}"],
                     "val_margin": r[f"val_margin_{comp}"], "seeds": r[f"seeds_{comp}"]})
    return pd.DataFrame(rows)


def gradient_tables(data: dict, c62: dict) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    names = data["names"]
    K = len(names)
    dj = [j["diag"] for j in data["jobs"] if j["diag"] is not None]
    if not dj:
        return pd.DataFrame(), pd.DataFrame(), np.zeros((0, 0, 0))
    shares = np.array([d["shares"] for d in dj])
    eff = np.array([d["eff_shares"] for d in dj])
    norms = np.array([d["norms"] for d in dj])
    cos = np.array([d["cos"] for d in dj])
    align = np.array([alignment_with_rest(d["norms"], d["cos"]) for d in dj])
    ratio = float(c62["grad_dominance_ratio"])
    rows = []
    for k, n in enumerate(names):
        s = float(shares[:, k].mean())
        rows.append({"group": data["group"], "task": n, "jobs": len(dj), "share": s, "share_sd": float(shares[:, k].std(ddof=1)) if len(dj) > 1 else float("nan"),
                     "share_x_balanced": s * K, "effective_share": float(eff[:, k].mean()), "norm_rel": float((norms[:, k] / norms.mean(1)).mean()),
                     "dominance": dominance_label(s, K, ratio), "align_with_rest": float(np.nanmean(align[:, k])) if K > 1 else float("nan")})
    pairs = pd.DataFrame(conflict_pairs(cos, names, float(c62["conflict_cos"]), float(c62["conflict_min_job_frac"]), float(c62["aligned_cos"])))
    if len(pairs):
        pairs.insert(0, "group", data["group"])
    return pd.DataFrame(rows), pairs, pad_stack([d["shares_epoch"] for d in dj])


def stopping_table(data: dict, c62: dict) -> pd.DataFrame:
    names = data["names"]
    rows = []
    per = {n: [] for n in names}
    for j in data["jobs"]:
        for k, n in enumerate(names):
            h = j["heads"][n]
            sr = stopping_regret(j["kinds"][k], h["hist_val_primary"], j["shared_epoch"])
            e = min(j["shared_epoch"], len(h["hist_val_primary"]) - 1)
            sr["progress"] = sr["err_shared"] / max(float(j["ref_err"][k]), 1e-8)
            sr["val_train_loss"] = float(h["hist_val_loss"][e] / max(h["hist_trsub_loss"][e], 1e-12))
            sr["n_train"] = int(h["n_train"])
            per[n].append(sr)
    mean_progress = float(np.mean([np.mean([r["progress"] for r in per[n]]) for n in names]))
    for n in names:
        rs = per[n]
        m = {k: float(np.mean([r[k] for r in rs])) for k in ("err_shared", "err_own", "regret", "regret_rel", "own_epoch", "progress", "val_train_loss")}
        rows.append({"group": data["group"], "task": n, "jobs": len(rs), "n_train_units": rs[0]["n_train"], "shared_epoch": float(np.mean([j["shared_epoch"] for j in data["jobs"]])),
                     "own_best_epoch": m["own_epoch"], "val_err_shared": m["err_shared"], "val_err_own_best": m["err_own"], "regret": m["regret"], "regret_rel": m["regret_rel"],
                     "stopping_compromise": bool(m["regret_rel"] >= float(c62["stopping_regret_rel"])), "progress_vs_head_only": m["progress"],
                     "lagging": bool(m["progress"] > mean_progress + float(c62["lag_margin"])), "val_loss_over_train_loss": m["val_train_loss"]})
    return pd.DataFrame(rows)


def cause_table(transfer: pd.DataFrame, grads: pd.DataFrame, pairs: pd.DataFrame, stop: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, t in transfer.iterrows():
        n = t["task"]
        g = grads[grads["task"] == n].iloc[0] if len(grads) and (grads["task"] == n).any() else None
        in_conflict = [] if not len(pairs) else [(r.head_b if r.head_a == n else r.head_a) for r in pairs.itertuples() if r.conflict and n in (r.head_a, r.head_b)]
        s = stop[stop["task"] == n].iloc[0] if (stop["task"] == n).any() else None
        flags = {"gradient": g["dominance"] if g is not None and g["dominance"] != "balanced" else "", "conflicts_with": ", ".join(in_conflict),
                 "stopping_compromise": bool(s["stopping_compromise"]) if s is not None else False, "lagging": bool(s["lagging"]) if s is not None else False}
        fired = [k for k, v in (("starved gradient" if flags["gradient"] == "starved" else "", flags["gradient"] == "starved"),
                                ("dominant gradient", flags["gradient"] == "dominant"), ("gradient conflict", bool(in_conflict)),
                                ("stopping compromise", flags["stopping_compromise"]), ("lagging", flags["lagging"])) if v]
        rows.append({"group": t["group"], "task": n, "transfer": t["transfer"], "delta": t["delta"], "val_flag": t["val_flag"], **flags,
                     "criteria_fired": "; ".join(fired) if fired else "none"})
    return pd.DataFrame(rows)


def ablation_table(spec_full: dict, abl_names: list[str], c62: dict, n_boot: int) -> pd.DataFrame:
    """Each finished SO762 ablation vs the full multi-task model and vs the matched single-task model, same seeds, paired speaker bootstrap."""
    bcfg = load_baseline_config()
    registry = build_tasks(bcfg)
    st_cfg = FT.load_config()
    full_cfg = MT.load_config(spec_full.get("variant"))
    full_tag = spec_full["tag"]
    floor = float(full_cfg["compare61"]["margin_floor"])
    seeds_all = sorted(set(full_cfg["seeds"]))
    prefix = c62["ablation_tag_prefix"]
    rows = []
    for abl in abl_names:
        acfg = MT.load_config(abl)
        tag = prefix + abl
        for head in acfg["groups"]["so762"]["tasks"]:
            task = registry[head]
            srcs = {"abl": [(acfg, tag)], "full": [(full_cfg, full_tag)], "st": [(st_cfg, "base")]}
            seeds = R61.common_seeds([R61.available(s, task, seeds_all) for s in srcs.values()])
            if not seeds:
                print(f"[us62] ablation {abl}/{head}: no finished jobs in common with the full model and the single-task model; skipped")
                continue
            y_all, meta = FH.load_labels(task)
            sysm = {k: R61.build_system(task, R61.load_sources(s, task, seeds), y_all, meta, n_boot, bcfg["bootstrap"]["seed"]) for k, s in srcs.items()}
            vs_full = R61.compare_head(task, sysm["full"], sysm["abl"], n_boot, floor)
            vs_st = R61.compare_head(task, sysm["st"], sysm["abl"], n_boot, floor)
            full_vs_st = R61.compare_head(task, sysm["st"], sysm["full"], n_boot, floor)
            rows.append({"ablation": abl, "task": head, "metric": sysm["abl"]["metric"], "seeds": len(seeds), "st": sysm["st"]["test"], "full_mt": sysm["full"]["test"],
                         "ablation_test": sysm["abl"]["test"], "full_vs_st": full_vs_st["delta"], "full_vs_st_verdict": full_vs_st["verdict"],
                         "abl_vs_full": vs_full["delta"], "abl_vs_full_lo": vs_full["lo"], "abl_vs_full_hi": vs_full["hi"], "abl_vs_full_p": vs_full["p"],
                         "abl_vs_full_verdict": vs_full["verdict"], "abl_vs_st": vs_st["delta"], "abl_vs_st_lo": vs_st["lo"], "abl_vs_st_hi": vs_st["hi"],
                         "abl_vs_st_p": vs_st["p"], "abl_vs_st_verdict": vs_st["verdict"], "val_flag_vs_st": vs_st["val_flag"],
                         "reading": ablation_reading(vs_st["verdict"], vs_full["verdict"])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def _fmt(v, nd=3):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (float, np.floating)):
        return "-" if not np.isfinite(v) else f"{v:.{nd}f}"
    return str(v)


def md_table(df: pd.DataFrame, cols: list[tuple[str, str]], nd: int = 3) -> list[str]:
    L = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        L.append("| " + " | ".join(_fmt(r[c], nd) for c, _ in cols) + " |")
    return L


def make_figures(group: str, names: list[str], cos_mean: np.ndarray, share_epoch: np.ndarray) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    FIG.mkdir(parents=True, exist_ok=True)
    short = [n.replace("phone_mispron_t1.0", "phone_mispron") for n in names]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    im = ax[0].imshow(cos_mean, vmin=-1, vmax=1, cmap="RdBu_r")
    ax[0].set_xticks(range(len(names)), short, rotation=60, ha="right")
    ax[0].set_yticks(range(len(names)), short)
    for i in range(len(names)):
        for j in range(len(names)):
            ax[0].text(j, i, f"{cos_mean[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax[0].set_title("mean gradient cosine (last shared layer)")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    m = np.nanmean(share_epoch, axis=0)
    for k, n in enumerate(short):
        ax[1].plot(range(m.shape[0]), m[:, k], marker="o", ms=3, label=n)
    ax[1].axhline(1.0 / len(names), color="k", ls=":", lw=1)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("share of summed gradient norms")
    ax[1].set_title("gradient share per head (dotted = balanced)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / f"us62_gradients_{group}.png", dpi=150)
    plt.close(fig)


def write_md(path, c62, per_group: dict, abl: pd.DataFrame | None) -> None:
    L = ["# US 6.2 - diagnosis of the multi-task models", "",
         "Everything below is descriptive and uses numbers that already exist (US 6.1 test comparison, the per-epoch gradient diagnostics logged during training, "
         "the per-head validation histories). The thresholds were fixed in `configs/multitask.yaml` (`compare62`) before the gradient diagnostics were looked at; "
         "the hypotheses about stopping epochs and about `phone_mispron_t1.0` came from the US 6.1 table, which had already been seen.", "",
         "Criteria: **negative transfer** = test verdict `worse` against the matched single-task model (same recipe), paired speaker bootstrap 95 % CI excludes 0 · "
         f"**dominant / starved** = gradient-norm share > {c62['grad_dominance_ratio']}/K or < 1/({c62['grad_dominance_ratio']}K) · "
         f"**conflict** = mean gradient cosine < {c62['conflict_cos']} in >= {c62['conflict_min_job_frac']:.0%} of the jobs · "
         f"**stopping compromise** = shared-checkpoint validation error >= {c62['stopping_regret_rel']:.0%} above the head's own best epoch · "
         f"**lagging** = relative progress over its head-only epoch worse than the head average by > {c62['lag_margin']}.", ""]
    for group, d in per_group.items():
        L += [f"## Group `{group}` (tag `{d['tag']}`)", "", "### 1. Transfer (multi-task minus matched single-task, test)", ""]
        L += md_table(d["transfer"], [("task", "head"), ("metric", "metric"), ("comparator", "comparator"), ("st_recipe", "ST recipe"), ("st", "single-task"), ("mt", "multi-task"),
                                      ("delta", "delta"), ("lo", "CI lo"), ("hi", "CI hi"), ("p", "p"), ("transfer", "transfer"), ("val_flag", "validation"), ("seeds", "seeds")])
        L += ["", "RMSE: negative delta favours multi-task. Macro-F1: positive delta favours multi-task.", ""]
        if len(d["grads"]):
            L += ["### 2. Task imbalance: gradient shares on the last shared transformer layer", ""]
            L += md_table(d["grads"], [("task", "head"), ("jobs", "jobs"), ("share", "share"), ("share_sd", "sd"), ("share_x_balanced", "x balanced"), ("effective_share", "share incl. loss weight"),
                                       ("dominance", "label"), ("align_with_rest", "cos with sum of others (approx.)")])
            L += ["", "Share = head's gradient norm / sum of all heads' norms (1/K = balanced). Norms are of the unweighted head losses; with the fixed equal weights of US 6.1 "
                  "the weighted share is the same. Different heads have different loss scales and unit counts, so a large share is a fact about the gradient, not a defect by itself.", "",
                  "### 3. Gradient conflict", ""]
            L += md_table(d["pairs"], [("head_a", "head A"), ("head_b", "head B"), ("mean_cos", "mean cos"), ("sd_cos", "sd over jobs"), ("min_cos", "min"), ("frac_jobs_below_thr", "jobs below threshold"),
                                       ("conflict", "conflict"), ("aligned", "aligned")])
            L += ["", f"![gradients]({'../../figures/multitask/us62_gradients_' + group + '.png'})", ""]
        L += ["### 4. Shared checkpoint vs each head's own best epoch, and progress", ""]
        L += md_table(d["stop"], [("task", "head"), ("n_train_units", "train units"), ("shared_epoch", "shared epoch"), ("own_best_epoch", "own best epoch"), ("val_err_shared", "val error (shared)"),
                                  ("val_err_own_best", "val error (own best)"), ("regret_rel", "regret (rel.)"), ("stopping_compromise", "compromise"), ("progress_vs_head_only", "error / head-only error"),
                                  ("lagging", "lagging"), ("val_loss_over_train_loss", "val loss / train loss")])
        L += ["", "Regret is an optimistic bound on what per-head checkpointing could gain: the best epoch of each head is chosen on the same validation data it is scored on.", "",
              "### 5. Which criteria fire for which head", ""]
        L += md_table(d["causes"], [("task", "head"), ("transfer", "transfer"), ("delta", "delta"), ("val_flag", "validation"), ("gradient", "gradient"), ("conflicts_with", "conflicts with"),
                                    ("stopping_compromise", "stopping compromise"), ("lagging", "lagging"), ("criteria_fired", "criteria fired")])
        L += [""]
    if abl is not None and len(abl):
        L += ["## Ablations (SO762)", "",
              "Same recipe (`base`), same seeds, same speakers. delta = ablation minus X on test; RMSE: negative favours the ablation; macro-F1: positive favours the ablation. "
              "`reading` is the pre-declared rule: " + " ".join(str(c62["ablation_reading"]).split()), ""]
        L += md_table(abl, [("ablation", "ablation"), ("task", "head"), ("metric", "metric"), ("st", "single-task"), ("full_mt", "full MT"), ("ablation_test", "ablation"),
                            ("abl_vs_full", "delta vs full MT"), ("abl_vs_full_lo", "lo"), ("abl_vs_full_hi", "hi"), ("abl_vs_full_verdict", "verdict"),
                            ("abl_vs_st", "delta vs single-task"), ("abl_vs_st_lo", "lo"), ("abl_vs_st_hi", "hi"), ("abl_vs_st_verdict", "verdict"), ("reading", "reading")])
        L += [""]
    L += ["## Caveats", "",
          "- One probe (the last trainable transformer layer). Conflicts that exist only in earlier layers are not seen; a consistent picture over seeds is the evidence, not one number.",
          "- Gradients are averaged over steps within each epoch; cosines of averaged gradients are not the average cosines. `cos with sum of others` is rebuilt from averaged norms and cosines (approximate).",
          "- SO762 has 19 validation speakers: a regret smaller than a few percent is inside the noise of the epoch choice. L2-ARCTIC has 4 folds x 3 seeds.",
          "- `phone_mispron_t1.0` is `phone_score < 1.0` and `sent_total` is correlated with the other sentence scores: near-duplicate targets make gradients look aligned without being independent evidence of transfer.",
          "- Ablations change the number of heads, which changes the stopping criterion (mean over the heads present); that is part of the intervention and is stated, not corrected.", ""]
    path.write_text("\n".join(L))


# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="*", choices=["so762", "l2"])
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--n-boot", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = MT.load_config()
    c62 = cfg["compare62"]
    registry = build_tasks(load_baseline_config())
    OUT.mkdir(parents=True, exist_ok=True)
    per_group, tabs = {}, {k: [] for k in ("transfer", "grads", "pairs", "stop", "causes")}
    for group, spec in c62["group_tags"].items():
        if a.groups and group not in a.groups:
            continue
        data = collect_group(group, spec["tag"], spec.get("variant"), registry)
        if data is None:
            print(f"[us62] {group}: no finished jobs under tag {spec['tag']!r}; skipped")
            continue
        tr = transfer_table(group, spec, data["names"])
        gr, pr, share_ep = gradient_tables(data, c62)
        st = stopping_table(data, c62)
        ca = cause_table(tr, gr, pr, st)
        if len(gr):
            cos_mean = np.mean([j["diag"]["cos"] for j in data["jobs"] if j["diag"] is not None], axis=0)
            make_figures(group, data["names"], cos_mean, share_ep)
        per_group[group] = {"tag": spec["tag"], "transfer": tr, "grads": gr, "pairs": pr, "stop": st, "causes": ca}
        for k, v in zip(tabs, (tr, gr, pr, st, ca)):
            tabs[k].append(v)
    if not per_group:
        raise SystemExit("nothing to report")
    for k, v in tabs.items():
        pd.concat([x for x in v if len(x)], ignore_index=True).to_csv(OUT / f"us62_{k}.csv", index=False)
    abl = None
    if not a.no_ablations and "so762" in per_group:
        abl = ablation_table(c62["group_tags"]["so762"], list(c62["ablations"]), c62, a.n_boot or cfg["compare61"]["n_boot"])
        if len(abl):
            abl.to_csv(OUT / "us62_ablations.csv", index=False)
    write_md(OUT / "us62_report.md", c62, per_group, abl)
    print((OUT / "us62_report.md").read_text())


if __name__ == "__main__":
    main()
