"""
US 4.3 - fixes for the weaknesses diagnosed in US 4.2 (frozen wav2vec 2.0 / HuBERT / WavLM encoders, same tasks / splits / labels).

    # fix 1: per-task layer re-selection from layers 6-12 (probe-ranked shortlist, then the full head protocol on the shortlisted layers)
    python -m src.models.frozen_fix layers   --encoder wav2vec2_base
    # fixes 2-4: recipes defined in configs/frozen.yaml (fix43.recipes): learned layer-weighted sum, sentence regularisation, word pooling
    python -m src.models.frozen_fix run      --encoder wavlm_base_plus --recipe ws --stage search
    python -m src.models.frozen_fix run      --encoder wavlm_base_plus --recipe ws --stage final
    python -m src.models.frozen_fix status   --encoder wavlm_base_plus --recipe ws
    # fix 5: out-of-fold probabilities of the mispronunciation head (train speakers) for per-age-group decision weights
    python -m src.models.frozen_fix mispron  --encoder wavlm_base_plus

Same protocol as US 4.1 (src/models/frozen_heads.py): standardisation fitted on the training rows only, hyper-parameters / early stopping
chosen on VALIDATION only, `search` = every grid point with one seed, `final` = the selected configs re-fitted with every seed and the
seed-averaged output used. Result files have the same layout and fields as frozen_heads, under ``fix43.root`` (default
experiments/results/frozen43) as ``<encoder>/<recipe>/<task>/<head>/<config>[__final]/fold<k>.npz``, so src/evaluation/report_frozen.py
helpers (pooled, apply_variants, select_config) read them unchanged.

What a recipe is (configs/frozen.yaml, fix43.recipes.<name>)
  layer   ws          learned softmax-weighted sum over layers ``fix43.ws_layers`` (inclusive), trained jointly with the head
          avgA-B      fixed mean of layers A..B (needed when PCA is used: PCA is fitted on a fixed input)
          <int>       one layer
  pool    base        the cached unit as is (default).  For so762_word: mean | max | std and '+'-joined combinations of them
  inputs  emb, side, pros   (side = Sprint 3 non-acoustic columns, pros = US 3.3 pitch / energy / pause columns)
  grids   optional override of heads.grids. Keys understood by the trainer: wd, dropout, hidden, in_drop (input dropout), pca (0 = off).
Layer-weighted sum: every layer is first standardised per dimension (train rows), then z = sum_l softmax(a)_l * x_l, then
[z, side, pros] go into the linear / MLP head. The layer logits ``a`` start at 0 (uniform average) and have no weight decay.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.ssl_embeddings import cache_root, tables_dir
from src.models import frozen_heads as FH
from src.models.baselines import REPO, Task, split_masks

SIDE_INPUTS = ("side", "pros")
HEADS = FH.HEADS


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def load_config() -> dict:
    return FH.load_config()


def fix_root(cfg: dict | None = None, root=None) -> Path:
    """Results root of the US 4.3 runs: ``root`` if given, else configs/frozen.yaml fix43.root."""
    if root:
        return Path(root)
    cfg = cfg or load_config()
    return REPO / cfg["fix43"].get("root", "experiments/results/frozen43")


def recipe_cfg(cfg: dict, name: str) -> tuple[dict, dict]:
    """(heads-style config for this recipe, the recipe spec). ``hcfg`` plugs into frozen_heads.head_grid / select_config."""
    fx = cfg["fix43"]
    if name not in fx["recipes"]:
        raise KeyError(f"unknown recipe {name!r}; configs/frozen.yaml fix43.recipes has {sorted(fx['recipes'])}")
    r = dict(fx["recipes"][name])
    h = dict(cfg["heads"])
    h["grids"] = r.get("grids", cfg["heads"]["grids"])
    h["tasks"] = r["tasks"]
    h["seeds"] = list(r.get("seeds", fx.get("seeds_final", h["seeds"])))
    h["mlp"] = {"hidden": int(r.get("hidden", h["mlp"]["hidden"]))}
    for k in ("layer", "pool", "inputs"):
        r.setdefault(k, {"layer": "ws", "pool": "base", "inputs": ["emb", "side"]}[k])
    return h, r


def recipe_base(encoder: str, recipe: str, root: Path | None = None) -> Path:
    return fix_root(root=root) / encoder / recipe


def parse_layer_spec(spec, ws_layers=(0, 12)) -> tuple[list[int], str]:
    """'ws' -> (all layers in ws_layers, 'ws'); 'avg6-12' -> ([6..12], 'avg'); 12 / '12' -> ([12], 'single')."""
    s = str(spec)
    if s == "ws":
        return list(range(int(ws_layers[0]), int(ws_layers[1]) + 1)), "ws"
    if s.startswith("avg"):
        a, b = (int(v) for v in s[3:].split("-"))
        return list(range(a, b + 1)), "avg"
    return [int(s)], "single"


# --------------------------------------------------------------------------- #
# data: embedding tensor (device, fp16) + standardised side / pros columns
# --------------------------------------------------------------------------- #
def _pool_parts(pool: str, table: str) -> list[str]:
    if pool in ("base", "mean") and "word" not in table:
        return ["base"]
    parts = ["mean" if p == "base" else p for p in pool.split("+")]
    if table != "so762_word" and parts != ["mean"]:
        raise ValueError(f"pool={pool!r} is only defined for the so762_word unit (got {table})")
    if not set(parts) <= {"mean", "max", "std"}:
        raise ValueError(f"unknown pool {pool!r}; use mean / max / std joined by '+'")
    return parts


def load_embedding(task: Task, encoder: str, layers: list[int], how: str, pool: str, n_rows: int, device: str,
                   cache: Path | None = None, chunk: int = 4096):
    """Torch tensor (n, Lk, D) float16 on ``device``: Lk = len(layers) for 'ws', 1 for 'avg' (mean of the layers) and 'single'."""
    import torch
    root = (cache or cache_root(load_config())) / encoder
    parts = _pool_parts(pool, task.table)
    E = np.load(root / f"{task.table}_X.npy", mmap_mode="r")
    M = np.load(root / f"{task.table}_ms_X.npy", mmap_mode="r") if {"max", "std"} & set(parts) else None
    assert E.shape[0] == n_rows, f"{task.table}: embeddings have {E.shape[0]} rows, Sprint 3 table has {n_rows}"
    assert max(layers) < E.shape[1], (layers, E.shape)
    if M is not None:
        assert M.shape[0] == n_rows and M.shape[1] == E.shape[1], f"{task.table}_ms cache {M.shape} does not match {E.shape}"
    D = E.shape[2]
    Lk = len(layers) if how == "ws" else 1
    width = D * len(parts) if parts != ["base"] else D
    out = torch.empty((n_rows, Lk, width), dtype=torch.float16, device=device)
    for a in range(0, n_rows, chunk):
        b = min(a + chunk, n_rows)
        blocks = []
        for p in parts:
            if p in ("base", "mean"):
                blocks.append(np.asarray(E[a:b])[:, layers, :])
            elif p == "max":
                blocks.append(np.asarray(M[a:b])[:, layers, :D])
            else:
                blocks.append(np.asarray(M[a:b])[:, layers, D:])
        x = np.concatenate(blocks, axis=-1).astype(np.float32)
        if how != "ws":
            x = x.mean(1, keepdims=True)
        assert np.isfinite(x).all(), f"{task.table}: non-finite cached embeddings in rows {a}:{b}"
        out[a:b] = torch.from_numpy(x.astype(np.float16)).to(device)
    return out


def load_extra(task: Task, kinds, n_rows: int, tdir: Path | None = None) -> np.ndarray | None:
    """Concatenated side / pros columns (float32, n x S) or None. Each file must have exactly the Sprint 3 rows."""
    tdir = tdir or tables_dir()
    parts = []
    for k in kinds:
        if k not in SIDE_INPUTS:
            continue
        A = np.load(tdir / f"{task.table}_{k}_X.npy").astype(np.float32)
        assert A.shape[0] == n_rows, f"{task.table}: {k} features have {A.shape[0]} rows, Sprint 3 table has {n_rows}"
        parts.append(A)
    if not parts:
        return None
    X = np.hstack(parts)
    assert np.isfinite(X).all(), f"{task.table}: non-finite side / pros features"
    return X


def build_data(task: Task, encoder: str, spec: dict, cfg: dict, device: str, cache: Path | None = None) -> dict:
    y, meta = FH.load_labels(task)
    layers, how = parse_layer_spec(spec["layer"], cfg["fix43"].get("ws_layers", [0, 12]))
    inputs = list(spec["inputs"])
    data = {"y": y, "meta": meta, "how": how, "layers": layers, "spec": spec, "device": device, "stats": {}, "pca": {}}
    data["E"] = load_embedding(task, encoder, layers, how, spec.get("pool", "base"), len(meta), device, cache) if "emb" in inputs else None
    X = load_extra(task, inputs, len(meta))
    data["X_raw"] = X
    return data


def _train_stats(data: dict, tr: np.ndarray, fold) -> dict:
    """Per-(layer, dim) mean / std of the embedding and per-column mean / std of side+pros, over TRAIN rows only (cached per fold)."""
    import torch
    key = (fold, len(tr), int(tr[0]), int(tr[-1]))
    if key in data["stats"]:
        return data["stats"][key]
    dev = data["device"]
    st = {}
    if data["E"] is not None:
        E = data["E"]
        s = torch.zeros(E.shape[1:], dtype=torch.float64, device=dev)
        ss = torch.zeros_like(s)
        idx = torch.as_tensor(tr, device=dev)
        for a in range(0, len(idx), 8192):
            x = E[idx[a:a + 8192]].double()
            s += x.sum(0)
            ss += (x * x).sum(0)
        mu = s / len(tr)
        sd = (ss / len(tr) - mu * mu).clamp_min(0).sqrt()
        sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
        st["mu"], st["sd"] = mu.float(), sd.float()
    if data["X_raw"] is not None:
        Xr = data["X_raw"]
        mu = Xr[tr].mean(0, dtype=np.float64).astype(np.float32)
        sd = Xr[tr].std(0, dtype=np.float64).astype(np.float32)
        sd[sd < 1e-6] = 1.0
        st["X"] = torch.as_tensor((Xr - mu) / sd, dtype=torch.float32, device=dev)
    data["stats"][key] = st
    return st


def _pca_components(data: dict, st: dict, tr: np.ndarray, kmax: int, fold) -> np.ndarray:
    """Top-``kmax`` principal axes (kmax x D) of the standardised TRAIN embeddings (only for a fixed, single-layer input)."""
    import torch
    key = (fold, len(tr), int(tr[0]), kmax)
    if key not in data["pca"]:
        E = data["E"]
        assert E.shape[1] == 1, "PCA needs a fixed layer input (layer: <int> or avgA-B), not the layer-weighted sum"
        idx = torch.as_tensor(tr, device=E.device)
        x = ((E[idx][:, 0].float() - st["mu"][0]) / st["sd"][0]).cpu().numpy().astype(np.float64)
        x -= x.mean(0)
        _, S, Vt = np.linalg.svd(x, full_matrices=False)
        k = min(kmax, Vt.shape[0])
        data["pca"][key] = (Vt[:k].astype(np.float32), float(S[0] / np.sqrt(len(tr))))
    return data["pca"][key]


# --------------------------------------------------------------------------- #
# model + trainer
# --------------------------------------------------------------------------- #
_MODEL_CLS = None


def _model_cls():
    global _MODEL_CLS
    if _MODEL_CLS is None:
        import torch
        import torch.nn as nn

        class Head(nn.Module):
            """[softmax-weighted layer sum of the standardised embedding, side / pros columns] -> linear / MLP head."""

            def __init__(self, n_layers, d_emb, d_extra, d_out, head, hp, ws):
                super().__init__()
                self.ws = ws
                self.alpha = nn.Parameter(torch.zeros(n_layers), requires_grad=ws)
                self.in_drop = nn.Dropout(float(hp.get("in_drop", 0.0)))
                self.head = FH.make_head(d_emb + d_extra, d_out, head, {**hp, "hidden": int(hp.get("hidden", 512))})

            def layer_weights(self):
                return torch.softmax(self.alpha, 0)

            def forward(self, E, X):
                z = (E * self.layer_weights()[None, :, None]).sum(1) if self.ws else E[:, 0]
                if X is not None:
                    z = torch.cat([z, X], 1)
                return self.head(self.in_drop(z))

        _MODEL_CLS = Head
    return _MODEL_CLS


def train_fix(data: dict, st: dict, yi: np.ndarray, tr: np.ndarray, va: np.ndarray, others: dict, *, kind: str, n_classes: int,
              head: str, hp: dict, seed: int, tcfg: dict, device: str, cw: np.ndarray | None = None, pca: tuple | None = None) -> dict:
    """Fit one head with early stopping on validation loss. ``others`` maps name -> row indices to predict.
    Returns {'val', *others} outputs (regression: original target scale; classification: softmax probabilities) + bookkeeping
    (+ 'ws_weights' for the layer-weighted sum)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    dev = torch.device(device)
    reg = kind == "reg"
    ws = data["how"] == "ws"
    E = data["E"]
    X = st.get("X")
    if pca is not None:                                       # replace the embedding by its (train-fitted) PCA projection, unwhitened
        V, scale = pca
        Vt = torch.as_tensor(V, device=dev)
        mu0, sd0 = st["mu"][0], st["sd"][0]
        Z = torch.empty((E.shape[0], 1, V.shape[0]), dtype=torch.float32, device=dev)
        for a in range(0, E.shape[0], 8192):
            Z[a:a + 8192, 0] = (((E[a:a + 8192, 0].float() - mu0) / sd0) @ Vt.T) / max(scale, 1e-6)
        E_use, mu, sd = Z, torch.zeros(1, V.shape[0], device=dev), torch.ones(1, V.shape[0], device=dev)
    else:
        E_use, mu, sd = E, st.get("mu"), st.get("sd")
    n_layers = 1 if E_use is None else E_use.shape[1]
    d_emb = 0 if E_use is None else E_use.shape[2]
    d_extra = 0 if X is None else X.shape[1]
    model = _model_cls()(n_layers, d_emb, d_extra, 1 if reg else n_classes, head, hp, ws and pca is None).to(dev)

    def get(idx_t):
        e = None
        if E_use is not None:
            e = E_use[idx_t].float()
            if pca is None:
                e = (e - mu) / sd
        else:
            e = torch.zeros((len(idx_t), 1, 0), device=dev)
        return e, (None if X is None else X[idx_t])

    if reg:
        m_y, s_y = float(yi[tr].mean()), float(yi[tr].std()) or 1.0
        y_t = torch.as_tensor((yi - m_y) / s_y, dtype=torch.float32, device=dev)
        loss_fn = nn.MSELoss()
    else:
        y_t = torch.as_tensor(yi, dtype=torch.long, device=dev)
        loss_fn = nn.CrossEntropyLoss(weight=None if cw is None else torch.as_tensor(cw, dtype=torch.float32, device=dev))
    head_params = [p for n_, p in model.named_parameters() if n_ != "alpha"]
    groups = [{"params": head_params, "weight_decay": hp["wd"]}]
    if ws and pca is None:
        groups.append({"params": [model.alpha], "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=tcfg["lr"])
    gen = torch.Generator().manual_seed(seed)
    tr_t, va_t = torch.as_tensor(tr, device=dev), torch.as_tensor(va, device=dev)
    n, bs = len(tr_t), tcfg["batch_size"]
    big = 4096 if ws else 16384

    def forward_idx(idx_t):
        outs = []
        for a in range(0, len(idx_t), big):
            e, x = get(idx_t[a:a + big])
            outs.append(model(e, x))
        out = torch.cat(outs)
        return out.squeeze(-1) if reg else out

    best, best_ep, bad = np.inf, 0, 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for ep in range(tcfg["max_epochs"]):
        model.train()
        perm = torch.randperm(n, generator=gen).to(dev)
        for i in range(0, n, bs):
            idx = tr_t[perm[i:i + bs]]
            e, x = get(idx)
            out = model(e, x)
            loss = loss_fn(out.squeeze(-1) if reg else out, y_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(loss_fn(forward_idx(va_t), y_t[va_t]))
        if vloss < best - 1e-6:
            best, best_ep, bad = vloss, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tcfg["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()

    def predict(idx):
        with torch.no_grad():
            out = forward_idx(torch.as_tensor(idx, device=dev))
            return (out * s_y + m_y).cpu().numpy() if reg else torch.softmax(out, 1).cpu().numpy()

    res = {"val": predict(va), **{k: predict(v) for k, v in others.items()}}
    res.update({"best_epoch": best_ep, "epochs_run": ep + 1, "val_loss": best})
    if ws and pca is None:
        res["ws_weights"] = model.layer_weights().detach().cpu().numpy()
    return res


# --------------------------------------------------------------------------- #
# jobs
# --------------------------------------------------------------------------- #
def fit_job_fix(task: Task, data: dict, hcfg: dict, head: str, config: str, hp: dict, fold: int, seeds: list[int], out: Path,
                device: str) -> dict:
    """Fit one config on one fold with every seed; write the seed-averaged predictions (frozen_heads.fit_job's file format)."""
    y, meta = data["y"], data["meta"]
    m = split_masks(task, meta, fold)
    K = None if task.kind == "reg" else len(task.labels)
    yi = FH.encode_y(task, y)
    tr, va, te = (np.flatnonzero(m[s]) for s in ("train", "val", "test"))
    st = _train_stats(data, tr, fold)
    sub = tr[np.random.default_rng(0).choice(len(tr), min(len(tr), hcfg["train"]["train_eval_rows"]), replace=False)]
    cw = FH.balanced_weights(yi[tr], K) if hp.get("cw") == "balanced" else None
    pca = None
    if int(hp.get("pca", 0)) > 0:
        pca = _pca_components(data, st, tr, int(hp["pca"]), fold)
    t0, outs, per_seed, best_eps = time.time(), [], [], []
    for s in seeds:
        r = train_fix(data, st, yi, tr, va, {"test": te, "train": sub}, kind=task.kind, n_classes=K or 1, head=head, hp=hp, seed=s,
                      tcfg=hcfg["train"], device=device, cw=cw, pca=pca)
        outs.append(r)
        best_eps.append(r["best_epoch"])
        per_seed.append(FH.primary_score(task, y[va], r["val"] if task.kind == "reg" else FH.decode_labels(task, r["val"].argmax(1))))
    mean = {k: np.mean([o[k] for o in outs], axis=0) for k in ("val", "test", "train")}
    extra = {}
    if task.kind == "reg":
        pred = dict(mean)
    else:
        pred = {k: FH.decode_labels(task, mean[k].argmax(1)) for k in mean}
        extra = {"val_proba": mean["val"].astype(np.float32), "test_proba": mean["test"].astype(np.float32),
                 "classes": np.array(task.labels).astype(str),
                 "train_proba": mean["train"].astype(np.float32), "train_y": yi[sub]}
    if "ws_weights" in outs[0]:
        extra["ws_weights"] = np.mean([o["ws_weights"] for o in outs], axis=0).astype(np.float32)
    train_score = FH.primary_score(task, y[sub], pred["train"])
    out.parent.mkdir(parents=True, exist_ok=True)
    d_emb = 0 if data["E"] is None else (pca[0].shape[0] if pca is not None else data["E"].shape[1] * data["E"].shape[2])
    n_feat = d_emb + (0 if data["X_raw"] is None else data["X_raw"].shape[1])
    np.savez_compressed(out, val_pred=pred["val"], test_pred=pred["test"], train_score=train_score, n_train=len(tr), n_features=n_feat,
                        seeds=np.array(seeds), val_score_seeds=np.array(per_seed), best_epochs=np.array(best_eps),
                        fit_s=time.time() - t0, **extra)
    val = float(np.mean(per_seed))
    return {"fit_s": time.time() - t0, "val": val if task.kind == "reg" else -val}


def run_recipe(encoder: str, recipe: str, stage: str, only_tasks=None, only_heads=None, budget: float | None = None,
               cache: Path | None = None, root: Path | None = None, force: bool = False) -> None:
    cfg = load_config()
    hcfg, spec = recipe_cfg(cfg, recipe)
    device = FH.pick_device(hcfg.get("device", "auto"))
    base = recipe_base(encoder, recipe, root)
    t_start, done, skipped = time.time(), 0, 0
    seeds_all = list(hcfg["seeds"])
    print(f"[fix43] {encoder} recipe={recipe} (layer={spec['layer']} pool={spec['pool']} inputs={spec['inputs']}) stage={stage} "
          f"device={device} -> {base}")
    for task in FH.main_tasks(None, hcfg):
        if only_tasks and task.name not in only_tasks:
            continue
        data = None
        for head in (only_heads or hcfg["heads"]):
            if stage == "search":
                todo = list(FH.head_grid(hcfg, task, head))
            else:
                y_, meta_ = FH.load_labels(task)
                todo = []
                for g in FH.groups_for(task):
                    sel = FH.select_config(base, task, hcfg, head, g, y_, meta_)
                    if sel is None:
                        print(f"[fix43] {task.name} {head} group={g}: search stage incomplete, skipping")
                        continue
                    todo.append((sel[1], sel[2]))
            for config, hp in todo:
                for fold in task.folds:
                    out = FH.job_path(base, task.name, head, config, fold, stage)
                    if out.exists() and not force:
                        skipped += 1
                        continue
                    if budget and time.time() - t_start > budget:
                        print(f"[fix43] budget reached; {done} jobs fitted this call - re-run to continue")
                        return
                    if data is None:
                        data = build_data(task, encoder, spec, cfg, device, cache)
                    info = fit_job_fix(task, data, hcfg, head, config, {**hp, "hidden": hp.get("hidden", hcfg["mlp"]["hidden"])}, fold,
                                       seeds_all if stage == "final" else seeds_all[:1], out, device)
                    done += 1
                    print(f"[fix43] {task.name:18s} {head:6s} {config:44s} fold{fold} {info['fit_s']:6.1f}s val={info['val']:.4f}", flush=True)
        del data
    print(f"[fix43] finished: {done} fitted, {skipped} already present")


def status(encoder: str, recipe: str, root: Path | None = None) -> None:
    cfg = load_config()
    hcfg, _ = recipe_cfg(cfg, recipe)
    base = recipe_base(encoder, recipe, root)
    rows = []
    for task in FH.main_tasks(None, hcfg):
        for head in hcfg["heads"]:
            for config, _ in FH.head_grid(hcfg, task, head):
                rows.append((task.name, "search", all(FH.job_path(base, task.name, head, config, f).exists() for f in task.folds)))
            rows += [(task.name, "final", all(FH.job_path(base, task.name, head, c, f, "final").exists() for f in task.folds))
                     for c in {p.name.replace("__final", "") for p in (base / task.name / head).glob("*__final")}]
    df = pd.DataFrame(rows, columns=["task", "stage", "done"])
    print(df.groupby(["task", "stage"])["done"].agg(["sum", "size"]).to_string())


# --------------------------------------------------------------------------- #
# fix 1: layer re-selection (layers 6-12), per task
# --------------------------------------------------------------------------- #
def probe_val_table(encoder: str, layers, tasks, mode: str = "emb+side", frozen_root: Path | None = None) -> pd.DataFrame:
    """Validation score of the per-layer linear probe (US 4.2, configs/frozen.yaml heads.probe) for ``tasks`` x ``layers``.
    ``score`` is lower-is-better (RMSE; negative macro-F1). Missing probe files are reported, not guessed."""
    cfg = load_config()
    pcfg = FH.probe_cfg(cfg["heads"])
    rows, missing = [], []
    for task in FH.main_tasks(None, {**pcfg, "tasks": tasks}):
        y, meta = FH.load_labels(task)
        group = "none" if task.kind == "reg" else "balanced"
        for layer in layers:
            base = FH.results_root(encoder, layer, mode, frozen_root)
            sel = FH.select_config(base, task, pcfg, "linear", group, y, meta)
            if sel is None:
                missing.append((task.name, layer))
                continue
            rows.append({"encoder": encoder, "task": task.name, "layer": int(layer), "score": float(sel[0]), "config": sel[1]})
    if missing:
        print(f"[fix43] WARNING: no probe results for {len(missing)} (task, layer) pairs, e.g. {missing[:4]}; run "
              f"`python -m src.models.frozen_heads run --encoder {encoder} --probe --layers 6-12 --tasks <task>` first")
    return pd.DataFrame(rows)


def shortlist_layers(encoder: str, tasks, window=(6, 12), k: int = 2, mode: str = "emb+side", frozen_root: Path | None = None) -> dict:
    """{task: [layers]}: the ``k`` best layers in ``window`` by probe VALIDATION score (ties: deeper layer)."""
    layers = list(range(window[0], window[1] + 1))
    df = probe_val_table(encoder, layers, tasks, mode, frozen_root)
    out = {}
    if df.empty:
        return out
    for task, g in df.groupby("task"):
        g = g.sort_values(["score", "layer"], ascending=[True, False])
        out[task] = [int(l) for l in g["layer"].head(k)]
    return out


def run_layers(encoder: str, k: int | None = None, stages=("search", "final"), budget: float | None = None, cache: Path | None = None,
               frozen_root: Path | None = None, tasks=None, dry_run: bool = False) -> None:
    """Fix 1: for every task, run the full US 4.1 head protocol (both heads, 3 seeds in `final`) at the probe-ranked top-k layers of 6-12.
    Results go to experiments/results/frozen/<encoder>/L<layer>/emb+side (same place as the layer-12 benchmark). Resumable."""
    cfg = load_config()
    fx = cfg["fix43"]
    k = k or fx.get("shortlist_k", 2)
    tasks = tasks or cfg["heads"]["tasks"]
    sl = shortlist_layers(encoder, tasks, tuple(fx.get("layer_window", [6, 12])), k, frozen_root=frozen_root)
    by_layer: dict[int, list[str]] = {}
    for task, ls in sl.items():
        for l in ls:
            by_layer.setdefault(l, []).append(task)
    print(f"[fix43] {encoder}: layer shortlist (probe validation, top {k} of {fx.get('layer_window', [6, 12])}): "
          + json.dumps({t: sl[t] for t in sorted(sl)}))
    print("[fix43] jobs: " + ", ".join(f"L{l}: {len(ts)} task(s)" for l, ts in sorted(by_layer.items())))
    if dry_run:
        return
    t_start = time.time()
    for layer in sorted(by_layer):
        for stage in stages:
            left = None if not budget else max(budget - (time.time() - t_start), 1.0)
            FH.run(encoder, layer, "emb+side", stage, only_tasks=by_layer[layer], budget=left, cache=cache, base=frozen_root)


# --------------------------------------------------------------------------- #
# fix 5: out-of-fold probabilities for per-group decision weights (SO762 phone mispronunciation)
# --------------------------------------------------------------------------- #
def mispron_task() -> Task:
    return FH.main_tasks(None, {"tasks": ["phone_mispron_t1.0"]})[0]


def find_mispron_source(encoder: str, frozen_root: Path | None = None, root: Path | None = None) -> dict:
    """The validation-best class-weighted (argmax) mispronunciation head of this encoder over every candidate input:
    frozen layers 6..12 (emb+side, US 4.1 / 4.3 fix 1) and the layer-weighted sum recipe ``ws`` (fix 2)."""
    cfg = load_config()
    task = mispron_task()
    y, meta = FH.load_labels(task)
    cands = []
    srcs = [("layer", l, FH.results_root(encoder, l, "emb+side", frozen_root), cfg["heads"]) for l in range(6, 13)]
    ws_h, _ = recipe_cfg(cfg, "ws")
    srcs.append(("ws", "ws", recipe_base(encoder, "ws", root), ws_h))
    for kind, tag, base, hcfg in srcs:
        for head in hcfg["heads"]:
            if "phone_mispron_t1.0" not in hcfg["tasks"]:
                continue
            sel = FH.select_config(base, task, hcfg, head, "balanced", y, meta)
            # only fully evaluated systems: layers 6-11 also hold single-config US 4.2 probe files, which are not candidates
            if sel is not None and all(FH.job_path(base, task.name, head, sel[1], f, "final").exists() for f in task.folds):
                cands.append({"score": sel[0], "kind": kind, "tag": tag, "head": head, "config": sel[1], "hp": sel[2], "base": base})
    if not cands:
        raise SystemExit(f"[fix43] no class-weighted phone_mispron_t1.0 results with a `final` stage for {encoder}; run the US 4.1 / fix 1 / fix 2 `final` stage first")
    return sorted(cands, key=lambda c: c["score"])[0]


def run_mispron_oof(encoder: str, n_folds: int = 5, seed: int = 0, cache: Path | None = None, frozen_root: Path | None = None,
                    root: Path | None = None, force: bool = False) -> Path:
    """Speaker-grouped out-of-fold probabilities (balanced class weights, the validation-selected hyper-parameters, one seed) for all
    TRAIN rows of SO762 phone mispronunciation. The validation / test probabilities of the same head come from its normal result file.
    Early stopping inside each fold uses a random 10 % of the fold's training speakers."""
    cfg = load_config()
    hcfg = cfg["heads"]
    task = mispron_task()
    src = find_mispron_source(encoder, frozen_root, root)
    out = fix_root(cfg, root) / encoder / "mispron" / "oof.npz"
    if out.exists() and not force:
        print(f"[fix43] {out} exists (use --force to redo)")
        return out
    device = FH.pick_device(hcfg.get("device", "auto"))
    spec = ({"layer": "ws", "pool": "base", "inputs": ["emb", "side"]} if src["kind"] == "ws"
            else {"layer": int(src["tag"]), "pool": "base", "inputs": ["emb", "side"]})
    print(f"[fix43] {encoder}: mispronunciation source = {src['kind']}:{src['tag']} {src['head']} {src['config']} (val score {src['score']:.4f})")
    data = build_data(task, encoder, spec, cfg, device, cache)
    y, meta = data["y"], data["meta"]
    yi = FH.encode_y(task, y)
    tr_all = np.flatnonzero((meta["split"] == "train").to_numpy())
    speakers = np.array(sorted(meta.loc[tr_all, "speaker"].unique()))
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(speakers[rng.permutation(len(speakers))], np.arange(len(speakers)) % n_folds))
    row_fold = meta.loc[tr_all, "speaker"].map(fold_of).to_numpy()
    proba = np.full((len(tr_all), len(task.labels)), np.nan, np.float32)
    hp = {**src["hp"], "hidden": src["hp"].get("hidden", hcfg["mlp"]["hidden"])}
    for f in range(n_folds):
        hold = tr_all[row_fold == f]
        fit_rows = tr_all[row_fold != f]
        fit_spk = np.array(sorted(meta.loc[fit_rows, "speaker"].unique()))
        stop_spk = set(rng.choice(fit_spk, max(2, len(fit_spk) // 10), replace=False))
        is_stop = meta.loc[fit_rows, "speaker"].isin(stop_spk).to_numpy()
        tr, va = fit_rows[~is_stop], fit_rows[is_stop]
        st = _train_stats(data, tr, ("oof", f))
        cw = FH.balanced_weights(yi[tr], len(task.labels)) if hp.get("cw") == "balanced" else None
        t0 = time.time()
        r = train_fix(data, st, yi, tr, va, {"hold": hold}, kind="clf", n_classes=len(task.labels), head=src["head"], hp=hp, seed=seed,
                      tcfg=hcfg["train"], device=device, cw=cw)
        proba[row_fold == f] = r["hold"]
        print(f"[fix43] oof fold {f + 1}/{n_folds}: {len(hold)} rows, {len(tr)} fit rows, {time.time() - t0:.1f}s", flush=True)
    assert np.isfinite(proba).all()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, rows=tr_all, proba=proba, fold=row_fold, y=yi[tr_all],
                        source=json.dumps({"kind": src["kind"], "tag": str(src["tag"]), "head": src["head"], "config": src["config"],
                                           "base": str(src["base"]), "n_folds": n_folds, "seed": seed}))
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "status", "layers", "mispron"])
    ap.add_argument("--encoder", required=True, choices=list(cfg["encoders"]))
    ap.add_argument("--recipe", default=None, choices=list(cfg["fix43"]["recipes"]))
    ap.add_argument("--stage", default="search", choices=["search", "final"])
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--heads", nargs="*", choices=HEADS)
    ap.add_argument("--budget", type=float, default=None, help="seconds; stop starting new jobs after this (resumable)")
    ap.add_argument("--k", type=int, default=None, help="layers: shortlist size per task (default fix43.shortlist_k)")
    ap.add_argument("--folds", type=int, default=5, help="mispron: number of speaker-grouped folds")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="layers: print the shortlist and the jobs it implies, run nothing")
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--results-root", default=None, help="fix43 results root (default fix43.root)")
    a = ap.parse_args(argv)
    cache = cache_root(cfg, a.cache_root)
    root = Path(a.results_root) if a.results_root else None
    if a.cmd == "layers":
        run_layers(a.encoder, a.k, budget=a.budget, cache=cache, tasks=a.tasks, dry_run=a.dry_run)
    elif a.cmd == "mispron":
        run_mispron_oof(a.encoder, a.folds, cache=cache, root=root, force=a.force)
    else:
        if not a.recipe:
            ap.error("--recipe is required for run / status")
        if a.cmd == "run":
            run_recipe(a.encoder, a.recipe, a.stage, a.tasks, a.heads, a.budget, cache, root, a.force)
        else:
            status(a.encoder, a.recipe, root)


if __name__ == "__main__":
    main()
