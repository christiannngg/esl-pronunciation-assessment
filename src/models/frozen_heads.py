"""
US 4.1 - lightweight task heads on FROZEN SSL embeddings (wav2vec 2.0 / HuBERT / WavLM), same tasks / splits / labels as Sprint 3.

    python -m src.models.frozen_heads run    --encoder wavlm_base_plus --stage search
    python -m src.models.frozen_heads run    --encoder wavlm_base_plus --stage final     # re-fits ONLY the val-selected configs with all seeds
    python -m src.models.frozen_heads status --encoder wavlm_base_plus
    python -m src.models.frozen_heads run    --encoder wavlm_base_plus --probe --layers 0-12   # fast per-layer linear probe (US 4.2)
    python -m src.models.frozen_heads run    --encoder wavlm_base_plus --mode emb              # ablations: emb | side (| emb+side main)

Runs on Colab (needs only the embedding cache, the *_meta tables, the *_side_X side features and data/splits); reporting is done by
src/evaluation/report_frozen.py on the machine that holds the Sprint 3 results.

Inputs per task (`--mode`):
  emb       frozen embedding of one layer (``--layer``; an int, or "a-b" = mean of layers a..b)       <- embedding-only ablation
  emb+side  embedding + the Sprint 3 non-acoustic columns (canonical phone one-hot, durations, word position, ...)   <- main setting
  side      side features only (what the heads can do without any acoustic embedding)
Heads: ``linear`` (one Linear layer = a linear probe) and ``mlp`` (one hidden layer, GELU, dropout). Same protocol as Sprint 3:
  * standardisation is fitted on the training rows only; hyper-parameters (and early-stopping epoch) are chosen on VALIDATION only
    (regression: RMSE; classification: macro-F1); test predictions are written at fit time and only read for the selected config;
  * classification is run with and without balanced class weights in the loss (config key ``cw``); US 3.3 decision weights are
    applied afterwards in the report;
  * `search` = one seed for every grid point; `final` = the selected configs re-fitted with every seed in ``heads.seeds`` and the
    seed-averaged prediction (regression) / probability (classification) is the system output.

One job = (encoder, layer, mode, task, head, config, fold); outputs go to
experiments/results/frozen/<encoder>/L<layer>/<mode>/<task>/<head>/<config>[__final]/fold<k>.npz
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.ssl_embeddings import cache_root, tables_dir
from src.models.baselines import REPO, Task, build_tasks, split_masks
from src.models.baselines import load_config as load_baseline_config

SPLIT_CSV = REPO / "data" / "splits" / "so762_speakers.csv"
MODES = ("emb", "emb+side", "side")
HEADS = ("linear", "mlp")


def load_config() -> dict:
    return yaml.safe_load((REPO / "configs" / "frozen.yaml").read_text())


def results_root(encoder: str, layer, mode: str, root: Path | None = None) -> Path:
    return (root or REPO / "experiments" / "results" / "frozen") / encoder / f"L{layer}" / mode


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_labels(task: Task, tdir: Path | None = None, split_csv: Path | None = None):
    """(y, meta) of the Sprint 3 table in Sprint 3 row order, WITHOUT the MFCC matrix (it is not shipped to Colab)."""
    tdir = tdir or tables_dir()
    meta = pd.read_csv(tdir / f"{task.table}_meta.csv.gz", dtype={"utt_id": str, "speaker": str})
    if task.dataset == "so762":      # official flag in the meta is overridden by the Sprint 2 train/val/test speakers
        sp = pd.read_csv(split_csv or SPLIT_CSV, dtype={"speaker": str}).set_index("speaker")["split"]
        meta["split"] = meta["speaker"].map(sp)
        assert meta["split"].notna().all() and set(meta["split"]) == {"train", "val", "test"}
    y = meta[task.target].to_numpy()
    if task.thr is not None:
        y = (y < task.thr).astype(int)
    return y, meta.reset_index(drop=True)


def parse_layer(spec) -> list[int]:
    s = str(spec)
    if "-" in s:
        a, b = map(int, s.split("-"))
        return list(range(a, b + 1))
    return [int(s)]


def load_features(task: Task, encoder: str, layer, mode: str, n_rows: int, cache: Path | None = None, tdir: Path | None = None) -> np.ndarray:
    assert mode in MODES, mode
    parts = []
    if mode != "side":
        E = np.load((cache or cache_root(load_config())) / encoder / f"{task.table}_X.npy", mmap_mode="r")
        assert E.shape[0] == n_rows, f"{task.table}: embeddings have {E.shape[0]} rows, Sprint 3 table has {n_rows}"
        ids = parse_layer(layer)
        assert max(ids) < E.shape[1], (ids, E.shape)
        emb = np.asarray(E[:, ids[0]], dtype=np.float32)
        for l in ids[1:]:
            emb += np.asarray(E[:, l], dtype=np.float32)
        parts.append(emb / len(ids))
    if mode != "emb":
        S = np.load((tdir or tables_dir()) / f"{task.table}_side_X.npy")
        assert S.shape[0] == n_rows, f"{task.table}: side features have {S.shape[0]} rows, Sprint 3 table has {n_rows}"
        parts.append(S.astype(np.float32))
    X = np.hstack(parts) if len(parts) > 1 else parts[0]
    assert np.isfinite(X).all(), f"{task.table}: non-finite features"
    return X


def standardise(Xtr: np.ndarray, *others: np.ndarray):
    mu = Xtr.mean(0, dtype=np.float64).astype(np.float32)
    sd = Xtr.std(0, dtype=np.float64).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    return [(Xtr - mu) / sd] + [(o - mu) / sd for o in others]


# --------------------------------------------------------------------------- #
# Heads
# --------------------------------------------------------------------------- #
def pick_device(name: str = "auto") -> str:
    import torch
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"


def make_head(d_in: int, d_out: int, head: str, hp: dict):
    import torch.nn as nn
    if head == "linear":
        return nn.Linear(d_in, d_out)
    return nn.Sequential(nn.Linear(d_in, hp["hidden"]), nn.GELU(), nn.Dropout(hp["dropout"]), nn.Linear(hp["hidden"], d_out))


def balanced_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.maximum(np.bincount(y, minlength=n_classes), 1)
    return len(y) / (n_classes * counts)


def train_head(Xtr, ytr, Xva, yva, others: dict, *, kind: str, n_classes: int, head: str, hp: dict, seed: int, tcfg: dict,
               device: str, cw: np.ndarray | None = None) -> dict:
    """Fit one head, early-stopped on validation loss. Returns outputs for 'val' and every key of ``others`` plus bookkeeping.
    Outputs: regression -> predictions on the original target scale; classification -> softmax probabilities (n, K)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    dev = torch.device(device)
    reg = kind == "reg"
    model = make_head(Xtr.shape[1], 1 if reg else n_classes, head, hp).to(dev)
    xtr, xva = torch.as_tensor(Xtr, device=dev), torch.as_tensor(Xva, device=dev)
    if reg:
        mu, sd = float(ytr.mean()), float(ytr.std()) or 1.0
        ytr_t = torch.as_tensor((ytr - mu) / sd, dtype=torch.float32, device=dev)
        yva_t = torch.as_tensor((yva - mu) / sd, dtype=torch.float32, device=dev)
        loss_fn = nn.MSELoss()
    else:
        ytr_t, yva_t = (torch.as_tensor(v, dtype=torch.long, device=dev) for v in (ytr, yva))
        loss_fn = nn.CrossEntropyLoss(weight=None if cw is None else torch.as_tensor(cw, dtype=torch.float32, device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=hp["wd"])
    gen = torch.Generator().manual_seed(seed)
    n, bs = len(xtr), tcfg["batch_size"]

    def forward(x):
        out = torch.cat([model(x[i:i + 16384]) for i in range(0, len(x), 16384)])
        return out.squeeze(-1) if reg else out

    best, best_ep, bad = np.inf, 0, 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}      # fallback if the validation loss never improves
    for ep in range(tcfg["max_epochs"]):
        model.train()
        perm = torch.randperm(n, generator=gen).to(dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            out = model(xtr[idx])
            loss = loss_fn(out.squeeze(-1) if reg else out, ytr_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(loss_fn(forward(xva), yva_t))
        if vloss < best - 1e-6:
            best, best_ep, bad = vloss, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tcfg["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()

    def predict(X):
        with torch.no_grad():
            out = forward(torch.as_tensor(X, device=dev))
            return (out * sd + mu).cpu().numpy() if reg else torch.softmax(out, 1).cpu().numpy()

    res = {"val": predict(Xva), **{k: predict(v) for k, v in others.items()}}
    res.update({"best_epoch": best_ep, "epochs_run": ep + 1, "val_loss": best})
    return res


# --------------------------------------------------------------------------- #
# Grid / jobs
# --------------------------------------------------------------------------- #
def head_grid(hcfg: dict, task: Task, head: str) -> list[tuple[str, dict]]:
    """[(config_id, hp)] for one head on one task kind. Grid keys come from configs/frozen.yaml (heads.grids)."""
    grid = dict(hcfg["grids"][task.kind][head])
    if head == "mlp":
        grid.setdefault("hidden", [hcfg["mlp"]["hidden"]])
    out = []
    for vals in itertools.product(*[grid[k] for k in sorted(grid)]):
        hp = dict(zip(sorted(grid), vals))
        out.append(("_".join(f"{k}{v:g}" if isinstance(v, (int, float)) else f"{k}{v}" for k, v in hp.items()), hp))
    return out


def job_path(base: Path, task: str, head: str, config: str, fold: int, stage: str = "search") -> Path:
    return base / task / head / (config + ("__final" if stage == "final" else "")) / f"fold{fold}.npz"


def primary_score(task: Task, y, pred) -> float:
    from src.evaluation.report_baseline import _score
    r = _score(task, y, pred)
    return r["rmse"] if task.kind == "reg" else -r["macro_f1"]       # lower is better for both


def decode_labels(task: Task, idx: np.ndarray) -> np.ndarray:
    return np.array(task.labels, dtype=object)[idx].astype(str) if isinstance(task.labels[0], str) else np.array(task.labels)[idx]


def encode_y(task: Task, y: np.ndarray) -> np.ndarray:
    if task.kind == "reg":
        return np.asarray(y, float)
    m = {l: i for i, l in enumerate(task.labels)}
    return np.array([m[v] for v in y])


def fit_job(task: Task, data: dict, hcfg: dict, head: str, config: str, hp: dict, fold: int, seeds: list[int], out: Path,
            device: str) -> dict:
    """Fits ``head`` (one config) on one fold with every seed in ``seeds``; writes the seed-averaged predictions to ``out``."""
    X, y, meta = data["X"], data["y"], data["meta"]
    m = split_masks(task, meta, fold)
    K = None if task.kind == "reg" else len(task.labels)
    yi = encode_y(task, y)
    tr, va, te = (np.flatnonzero(m[s]) for s in ("train", "val", "test"))
    Xtr, Xva, Xte = standardise(X[tr], X[va], X[te])
    sub = np.random.default_rng(0).choice(len(tr), min(len(tr), hcfg["train"]["train_eval_rows"]), replace=False)
    cw = balanced_weights(yi[tr], K) if hp.get("cw") == "balanced" else None
    t0, outs, per_seed, best_eps = time.time(), [], [], []
    for s in seeds:
        r = train_head(Xtr, yi[tr], Xva, yi[va], {"test": Xte, "train": Xtr[sub]}, kind=task.kind, n_classes=K or 1, head=head, hp=hp,
                       seed=s, tcfg=hcfg["train"], device=device, cw=cw)
        outs.append(r)
        best_eps.append(r["best_epoch"])
        per_seed.append(primary_score(task, y[va], r["val"] if task.kind == "reg" else decode_labels(task, r["val"].argmax(1))))
    mean = {k: np.mean([o[k] for o in outs], axis=0) for k in ("val", "test", "train")}
    extra = {}
    if task.kind == "reg":
        pred = {k: mean[k] for k in mean}
    else:
        pred = {k: decode_labels(task, mean[k].argmax(1)) for k in mean}
        extra = {"val_proba": mean["val"].astype(np.float32), "test_proba": mean["test"].astype(np.float32),
                 "classes": np.array(task.labels).astype(str),
                 "train_proba": mean["train"].astype(np.float32), "train_y": yi[tr][sub]}      # lets the report score train with the SAME decision rule
    train_score = primary_score(task, y[tr][sub], pred["train"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, val_pred=pred["val"], test_pred=pred["test"], train_score=train_score, n_train=len(tr), n_features=X.shape[1],
                        seeds=np.array(seeds), val_score_seeds=np.array(per_seed), best_epochs=np.array(best_eps),
                        fit_s=time.time() - t0, **extra)
    val = float(np.mean(per_seed))
    return {"fit_s": time.time() - t0, "val": val if task.kind == "reg" else -val}      # RMSE / macro-F1 as printed by `run`


def select_config(base: Path, task: Task, hcfg: dict, head: str, group: str, y, meta, stage: str = "search"):
    """Best config of one cw-group ('none' / 'balanced') by mean per-fold VALIDATION score. Never opens test predictions."""
    best = None
    for config, hp in head_grid(hcfg, task, head):
        if task.kind == "clf" and hp.get("cw", "none") != group:
            continue
        scores = []
        for fold in task.folds:
            p = job_path(base, task.name, head, config, fold, stage)
            if not p.exists():
                scores = None
                break
            d = np.load(p, allow_pickle=True)
            scores.append(primary_score(task, y[split_masks(task, meta, fold)["val"]], d["val_pred"]))
        if scores is None:
            continue
        s = float(np.mean(scores))
        if best is None or s < best[0]:
            best = (s, config, hp)
    return best          # (score (lower = better), config, hp) or None


def groups_for(task: Task) -> list[str]:
    return ["none"] if task.kind == "reg" else ["none", "balanced"]


def main_tasks(cfg: dict, hcfg: dict) -> list[Task]:
    tasks = build_tasks(load_baseline_config())
    return [tasks[n] for n in hcfg["tasks"]]


def parse_layers(spec: str) -> list[int]:
    """'0-12' -> 0..12, '3,6,9' -> [3,6,9], '2-4,12' -> [2,3,4,12]."""
    out = []
    for part in str(spec).split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]
    return out


def probe_cfg(hcfg: dict) -> dict:
    """The fast per-layer probe is the same job set as the main search restricted to one linear config per task kind, so every
    layer-12 probe job already exists after the main run (same config ids, same files)."""
    return {**hcfg, "grids": hcfg["probe"]["grids"], "tasks": hcfg["probe"]["tasks"], "heads": ["linear"]}


def run(encoder: str, layer, mode: str, stage: str, only_tasks=None, only_heads=None, budget: float | None = None,
        cache: Path | None = None, base: Path | None = None, probe: bool = False, force: bool = False) -> None:
    cfg = load_config()
    hcfg = probe_cfg(cfg["heads"]) if probe else cfg["heads"]
    assert not (probe and stage != "search"), "the layer probe is a search-stage job set (one seed)"
    device = pick_device(hcfg.get("device", "auto"))
    base = results_root(encoder, layer, mode, base)
    t_start, done, skipped = time.time(), 0, 0
    seeds_all = list(hcfg["seeds"])
    print(f"[heads] {encoder} layer={layer} mode={mode} stage={stage}{' probe' if probe else ''}{' FORCE' if force else ''} device={device} -> {base}")
    for task in main_tasks(cfg, hcfg):
        if only_tasks and task.name not in only_tasks:
            continue
        data = None
        for head in (only_heads or hcfg["heads"]):
            todo = []
            if stage == "search":
                todo = [(c, hp) for c, hp in head_grid(hcfg, task, head)]
            else:
                y_, meta_ = load_labels(task)
                for g in groups_for(task):
                    sel = select_config(base, task, hcfg, head, g, y_, meta_)
                    if sel is None:
                        print(f"[heads] {task.name} {head} group={g}: search stage incomplete, skipping")
                        continue
                    todo.append((sel[1], sel[2]))
            for config, hp in todo:
                for fold in task.folds:
                    out = job_path(base, task.name, head, config, fold, stage)
                    if out.exists() and not force:
                        skipped += 1
                        continue
                    if budget and time.time() - t_start > budget:
                        print(f"[heads] budget reached; {done} jobs fitted this call - re-run to continue")
                        return
                    if data is None:
                        y, meta = load_labels(task)
                        data = {"y": y, "meta": meta, "X": load_features(task, encoder, layer, mode, len(meta), cache)}
                    info = fit_job(task, data, hcfg, head, config, {**hp, "hidden": hp.get("hidden", hcfg["mlp"]["hidden"])}, fold,
                                   seeds_all if stage == "final" else seeds_all[:1], out, device)
                    done += 1
                    print(f"[heads] {task.name:18s} {head:6s} {config:32s} fold{fold} {info['fit_s']:6.1f}s val={info['val']:.4f}", flush=True)
    print(f"[heads] finished: {done} fitted, {skipped} already present")


def status(encoder: str, layer, mode: str, base: Path | None = None) -> None:
    cfg = load_config()
    hcfg = cfg["heads"]
    base = results_root(encoder, layer, mode, base)
    rows = []
    for task in main_tasks(cfg, hcfg):
        for head in hcfg["heads"]:
            for config, _ in head_grid(hcfg, task, head):
                rows.append((task.name, "search", all(job_path(base, task.name, head, config, f).exists() for f in task.folds)))
            rows += [(task.name, "final", all(job_path(base, task.name, head, c, f, "final").exists() for f in task.folds))
                     for c in {p.parent.name.replace("__final", "") for p in (base / task.name / head).glob("*__final")}]
    df = pd.DataFrame(rows, columns=["task", "stage", "done"])
    print(df.groupby(["task", "stage"])["done"].agg(["sum", "size"]).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "status"])
    ap.add_argument("--encoder", required=True, choices=list(load_config()["encoders"]))
    ap.add_argument("--layer", default=None, help="int or 'a-b' (mean of layers); default heads.primary_layer")
    ap.add_argument("--layers", default=None, help="with --probe: sweep these layers, 'a-b' range or comma list (default 0-12)")
    ap.add_argument("--probe", action="store_true", help="fast per-layer linear probe (one config, one seed; configs/frozen.yaml heads.probe)")
    ap.add_argument("--force", action="store_true", help="re-fit jobs that already have a result file (e.g. to add the train probabilities)")
    ap.add_argument("--mode", default="emb+side", choices=MODES)
    ap.add_argument("--stage", default="search", choices=["search", "final"])
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--heads", nargs="*", choices=HEADS)
    ap.add_argument("--budget", type=float, default=None, help="seconds; stop starting new jobs after this (resumable)")
    ap.add_argument("--cache-root", default=None)
    a = ap.parse_args()
    layer = a.layer if a.layer is not None else load_config()["heads"]["primary_layer"]
    cache = cache_root(load_config(), a.cache_root)
    if a.cmd == "run":
        if a.probe and not a.layers:
            a.layers = "0-12"
        layers = parse_layers(a.layers) if a.layers else [layer]
        for l in layers:
            run(a.encoder, l, a.mode, a.stage, a.tasks, a.heads, a.budget, cache, probe=a.probe, force=a.force)
    else:
        status(a.encoder, layer, a.mode)
