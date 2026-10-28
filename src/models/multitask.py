"""
US 6.1 - shared-encoder multi-task fine-tuning with one head per task.

    python -m src.models.multitask dry-run                                  # build the groups, check that every head sees the same utterances / splits (no GPU)
    python -m src.models.multitask run --tag mt_base                        # both groups, all folds, all seeds (resumable; skips finished jobs)
    python -m src.models.multitask run --tag mt_lr --variant lr             # the US 5.3 `lr` recipe (configs/finetune.yaml variants), shared encoder
    python -m src.models.multitask run --tag t --groups so762 --folds 0 --seeds 0 --epochs 2 --force      # smoke test
    python -m src.models.multitask status --tag mt_base

Two multi-task models (configs/multitask.yaml `groups`):
    so762 : sent_accuracy, sent_fluency, sent_prosodic, sent_total, word_accuracy, phone_score, phone_mispron_t1.0   (7 heads, 1 fold)
    l2    : l2_phone_error, l2_word_error                                                                                (2 heads, 4 folds)

What is identical to the single-task fine-tuning of Sprint 5 (src/models/finetune.py), so the ONLY difference is that the encoder is shared:
  * encoder checkpoint, CNN front end frozen, layer-weighted sum, MLP head, optimiser groups, learning-rate schedule, head-only first epoch, early stopping
    patience, grad clip, bf16, batching (whole utterances, length-bucketed), exact batch-size-1 predictions for the headline numbers;
  * tasks, labels, rows and their order, splits, pooling spans and side features (the very same ``TaskData`` objects);
  * per-head loss: MSE on the train-standardised target (regression) or balanced class-weighted cross-entropy (classification).
What differs:
  * one forward pass of the encoder per utterance serves every head (sentence heads pool the whole utterance, word / phone heads pool MFA intervals);
  * the heads' losses are combined (``make_weighter``; US 6.1 = fixed, equal weights normalised to sum 1; US 6.3 adds `uncertainty` and `gradnorm`);
  * ONE checkpoint per multi-task job, picked on validation by ``train.stop_metric = mean_rel`` (see ``mean_rel``);
  * gradient diagnostics for US 6.2: every ``diag.every_steps`` optimiser steps, each head's gradient on the probe parameters (last transformer layer) is
    measured -> per-head gradient norm, its share of the total, and the cosine similarity between heads.

Outputs. One job = (tag, group, fold, seed):
    <root>/<tag>/<head>/fold<k>/seed<s>.npz      per head, SAME keys/layout as the single-task jobs (so src/evaluation/report_us51.py reads them unchanged);
                                                 extra keys: mt_group, mt_n_heads, mt_group_fit_s, mt_head_best_epoch
    <root>/<tag>/_group_<group>/fold<k>/seed<s>.npz   per-epoch group history: mean_rel, per-head gradient norms / cosines, effective loss weights
    <ckpt_root>/<tag>/<group>/fold<k>_seed<s>.pt      best checkpoint (shared encoder + all heads), fp16
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features import ssl_embeddings as se
from src.models import finetune as FT
from src.models import frozen_heads as FH
from src.models.baselines import REPO, Task, build_tasks
from src.models.baselines import load_config as load_baseline_config

SR = se.SR
CFG_PATH = REPO / "configs" / "multitask.yaml"


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def load_config(variant: str | None = None, path: Path | None = None) -> dict:
    """finetune.yaml (encoder, model, train) + multitask.yaml deltas + optional variant (multitask.yaml first, then finetune.yaml)."""
    path = Path(path or CFG_PATH)
    mt = yaml.safe_load(path.read_text())
    base = yaml.safe_load((path.parent / mt.get("inherit", "finetune.yaml")).read_text())
    cfg = {k: base[k] for k in ("seed", "encoder", "model", "train") if k in base}
    cfg = FT.deep_update(cfg, {k: v for k, v in mt.items() if k not in ("inherit", "variants")})
    cfg["variants"] = mt.get("variants", {})
    if variant:
        if variant in mt.get("variants", {}):
            over = mt["variants"][variant]
        elif variant in base.get("variants", {}):
            over = base["variants"][variant]
        else:
            raise KeyError(f"unknown variant {variant!r}; multitask.yaml defines {sorted(mt.get('variants', {}))}, finetune.yaml {sorted(base.get('variants', {}))}")
        cfg = FT.deep_update(cfg, over)
    return cfg


def job_file(cfg: dict, tag: str, task: str, fold: int, seed: int) -> Path:
    return FT.job_file(cfg, tag, task, fold, seed)


def group_file(cfg: dict, tag: str, group: str, fold: int, seed: int) -> Path:
    return REPO / cfg["root"] / tag / f"_group_{group}" / f"fold{fold}" / f"seed{seed}.npz"


def ckpt_file(cfg: dict, tag: str, group: str, fold: int, seed: int) -> Path:
    return REPO / cfg["ckpt_root"] / tag / group / f"fold{fold}_seed{seed}.pt"


def head_key(name: str) -> str:
    """nn.ModuleDict keys cannot contain '.' (``phone_mispron_t1.0``)."""
    return name.replace(".", "_")


def seeds_for_group(cfg: dict, tasks: dict[str, Task], group: str, override: list[int] | None = None) -> list[int]:
    first = tasks[cfg["groups"][group]["tasks"][0]]
    return list(override) if override else list(cfg["seeds"] if first.dataset == "so762" else cfg["l2_seeds"])


# --------------------------------------------------------------------------- #
# pure helpers (numpy only, unit-tested)
# --------------------------------------------------------------------------- #
def normalise_weights(names: list[str], weights: dict | None) -> dict[str, float]:
    """Head weights (missing = 1) normalised to sum 1."""
    w = {n: float((weights or {}).get(n, 1.0)) for n in names}
    if any(v <= 0 for v in w.values()):
        raise ValueError(f"loss weights must be positive: {w}")
    s = sum(w.values())
    return {n: v / s for n, v in w.items()}


def error_from_primary(kind: str, primary: float) -> float:
    """FH.primary_score is RMSE (regression) or -macro-F1 (classification), lower is better; the error is RMSE or 1 - macro-F1."""
    return float(primary) if kind == "reg" else 1.0 + float(primary)


def mean_rel(errors: dict[str, float], ref: dict[str, float]) -> float:
    """Mean over heads of error / reference error (reference = the head-only epoch 0). Scale free, lower is better, 1.0 = no better than the probe."""
    return float(np.mean([errors[n] / max(ref[n], 1e-8) for n in errors]))


def cosine_from_gram(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(norms, cosine matrix) from the Gram matrix of the heads' gradients. A head with a zero gradient has norm 0 and cosine 0 with everything."""
    G = np.asarray(G, float)
    n = np.sqrt(np.clip(np.diag(G), 0.0, None))
    denom = np.outer(n, n)
    cos = np.divide(G, denom, out=np.zeros_like(G), where=denom > 0)
    return n, np.clip(cos, -1.0, 1.0)


def grad_shares(norms: np.ndarray) -> np.ndarray:
    """Each head's share of the summed gradient norms (1/K each = balanced)."""
    norms = np.asarray(norms, float)
    s = norms.sum()
    return norms / s if s > 0 else np.full(len(norms), 1.0 / max(len(norms), 1))


# --------------------------------------------------------------------------- #
# loss weighting (US 6.1 = fixed; US 6.3 adds `uncertainty` and `gradnorm` behind the same interface)
# --------------------------------------------------------------------------- #
class FixedWeights:
    """total = sum_k w_k * loss_k with fixed w (normalised to sum 1). No parameters."""
    scheme = "fixed"
    needs_grad_norms = False

    def __init__(self, names: list[str], weights: dict | None = None):
        self.names = list(names)
        self.w = normalise_weights(self.names, weights)

    def parameters(self) -> list:
        return []

    # the loop calls these on every weighter; only GradNorm does anything
    def observe(self, losses: dict) -> None:
        pass

    def wants_norms(self, k: int) -> bool:
        return False

    def update(self, losses: dict, norms) -> None:
        pass

    def combine(self, losses: dict):
        """-> (scalar total loss tensor, {head: float weight actually applied})."""
        total = None
        for n in self.names:
            term = self.w[n] * losses[n]
            total = term if total is None else total + term
        return total, dict(self.w)


class UncertaintyWeights:
    """Learned homoscedastic-uncertainty weights (Kendall, Gal & Cipolla 2018), US 6.3.

    One learnable log-variance s_k per head (init 0). The weight of head k is c_k * exp(-s_k) with c = 1/2 for a regression head and 1 for a classification head;
    s_k is trained on  sum_k [ c_k exp(-s_k) L_k + s_k / 2 ]  (the L_k are detached there, so that term only moves s). The shared encoder and the heads receive
    the SAME combination normalised to sum 1 (detached), exactly as `fixed`, so the gradient clip and Adam see the scale of a single-task job and only the
    RELATIVE weights are learned. A head whose training loss stays high gets a large variance and a small weight; a head that fits its training data
    (loss -> 0) gets a large weight. s is clipped to [-S_CLIP, S_CLIP] inside the weights.
    """
    scheme = "uncertainty"
    needs_grad_norms = False
    S_CLIP = 5.0

    def __init__(self, names: list[str], kinds: dict, device=None):
        torch = FT._torch()
        self.names = list(names)
        self.c = torch.tensor([0.5 if kinds.get(n, "reg") == "reg" else 1.0 for n in self.names], dtype=torch.float32, device=device)
        self.s = torch.nn.Parameter(torch.zeros(len(self.names), dtype=torch.float32, device=device))

    def parameters(self) -> list:
        return [self.s]

    def observe(self, losses: dict) -> None:
        pass

    def wants_norms(self, k: int) -> bool:
        return False

    def update(self, losses: dict, norms) -> None:
        pass

    def combine(self, losses: dict):
        torch = FT._torch()
        s = self.s.clamp(-self.S_CLIP, self.S_CLIP)
        raw = self.c * torch.exp(-s)
        wn = (raw / raw.sum()).detach()
        L = torch.stack([losses[n] for n in self.names])
        total = (wn * L).sum() + (raw * L.detach()).sum() + 0.5 * s.sum()
        return total, {n: float(wn[i]) for i, n in enumerate(self.names)}


class GradNormWeights:
    """GradNorm (Chen et al. 2018), US 6.3, first-order form, numpy state. No learnable parameters in the graph.

    Weights w_k (mean 1, floor w_min) move so that each head's weighted gradient norm  G_k = w_k * ||grad_W L_k||  (W = the probe parameters, the last
    trainable transformer layer, as in the US 6.2 diagnostics) approaches  mean(G) * r_k**alpha,  where r_k is the head's relative inverse training rate,
    (L_k / L0_k) / mean_j(L_j / L0_j). L is an exponential moving average of the head's training loss, L0 its mean over the second half of the head-only
    epoch. The weights are updated by Adam (lr_w) on the GradNorm L1 loss with the targets held constant (the gradient w.r.t. w_k is sign(G_k - target_k) *
    ||grad_W L_k||), then floored and renormalised to mean 1. The model receives w / K (sum 1), as in `fixed`.
    """
    scheme = "gradnorm"
    needs_grad_norms = True

    def __init__(self, names: list[str], alpha: float = 0.5, lr: float = 0.025, update_every: int = 1, ema: float = 0.9, w_min: float = 0.05):
        self.names = list(names)
        self.K = len(self.names)
        self.alpha, self.lr, self.every, self.ema, self.w_min = float(alpha), float(lr), max(int(update_every), 1), float(ema), float(w_min)
        self.w = np.ones(self.K)
        self.L0 = None
        self.Lema = None
        self._warm: list[np.ndarray] = []
        self._m, self._v, self._t = np.zeros(self.K), np.zeros(self.K), 0

    def parameters(self) -> list:
        return []

    def observe(self, losses: dict) -> None:
        """Called every step with the unweighted head losses: collects the head-only reference, then smooths the loss."""
        vals = np.array([float(losses[n].detach()) for n in self.names])
        if self.L0 is None:
            self._warm.append(vals)
        else:
            self.Lema = self.ema * self.Lema + (1.0 - self.ema) * vals

    def wants_norms(self, k: int) -> bool:
        return k % self.every == 0

    def update(self, losses: dict, norms) -> dict:
        """norms: per-head gradient norm on the probe parameters of the UNWEIGHTED head losses. -> {'G': ..., 'target': ...} (diagnostic)."""
        norms = np.asarray(norms, float)
        if self.L0 is None:                                              # first update = first step with the encoder training
            warm = np.array(self._warm)
            self.L0 = np.maximum(warm[len(warm) // 2:].mean(0), 1e-8)
            self.Lema = warm[-1].copy()
        r = np.maximum(self.Lema / self.L0, 1e-8)
        r = r / r.mean()
        G = self.w * norms
        target = G.mean() * r ** self.alpha
        grad = np.sign(G - target) * norms
        self._t += 1
        self._m = 0.9 * self._m + 0.1 * grad
        self._v = 0.999 * self._v + 0.001 * grad ** 2
        step = (self._m / (1 - 0.9 ** self._t)) / (np.sqrt(self._v / (1 - 0.999 ** self._t)) + 1e-8)
        w = np.maximum(self.w - self.lr * step, self.w_min)
        self.w = w * (self.K / w.sum())
        return {"G": G, "target": target}

    def combine(self, losses: dict):
        wn = self.w / self.K
        total = None
        for i, n in enumerate(self.names):
            term = float(wn[i]) * losses[n]
            total = term if total is None else total + term
        return total, {n: float(wn[i]) for i, n in enumerate(self.names)}


def make_weighter(lcfg: dict, names: list[str], kinds: dict[str, str], device=None):
    lcfg = lcfg or {}
    scheme = lcfg.get("scheme", "fixed")
    if scheme == "fixed":
        return FixedWeights(names, lcfg.get("weights"))
    if scheme == "uncertainty":
        return UncertaintyWeights(names, kinds, device)
    if scheme == "gradnorm":
        return GradNormWeights(names, alpha=float(lcfg.get("alpha", 0.5)), lr=float(lcfg.get("lr_w", 0.025)), update_every=int(lcfg.get("update_every", 1)),
                               ema=float(lcfg.get("ema", 0.9)), w_min=float(lcfg.get("w_min", 0.05)))
    raise ValueError(f"unknown loss scheme {scheme!r} (fixed | uncertainty | gradnorm)")


# --------------------------------------------------------------------------- #
# data: the heads of one group over the SAME utterances
# --------------------------------------------------------------------------- #
class MTData:
    """The per-task ``TaskData`` objects of one group, aligned on a canonical utterance list (the first head's)."""

    def __init__(self, tasks: list[Task], do_normalize: bool, l2_half_s: float | None = None, split_csv: Path | None = None):
        self.names = [t.name for t in tasks]
        self.tasks = {t.name: t for t in tasks}
        self.td = {t.name: FT.TaskData(t, do_normalize, l2_half_s, split_csv) for t in tasks}
        self.main = self.td[self.names[0]]
        self.utt_ids = self.main.utt_ids
        self.map: dict[str, np.ndarray] = {}                       # head -> (canonical utterance -> that head's utterance index)
        for n in self.names:
            td = self.td[n]
            pos = {u: i for i, u in enumerate(td.utt_ids)}
            m = np.array([pos.get(u, -1) for u in self.utt_ids], np.int64)
            if len(td.utt_ids) != len(self.utt_ids) or (m < 0).any():
                raise ValueError(f"{n}: does not label exactly the utterances of {self.names[0]}")
            if not (td.utt_speaker[m] == self.main.utt_speaker).all():
                raise ValueError(f"{n}: speaker of some utterance differs from {self.names[0]}")
            self.map[n] = m
        folds = {tuple(t.folds) for t in tasks}
        if len(folds) != 1:
            raise ValueError(f"heads of one group must share their folds, got {folds}")
        self.folds = list(self.tasks[self.names[0]].folds)
        self.table_head: dict[str, str] = {}                       # table -> representative head (spans / rows are identical for heads on one table)
        for n in self.names:
            self.table_head.setdefault(self.tasks[n].table, n)
        self.table_kind = {tb: self.td[h].pool_kind for tb, h in self.table_head.items()}
        self._splits: dict[int, dict] = {}

    # ---- audio ----
    @property
    def lengths(self) -> np.ndarray:
        return self.main.lengths

    def wav(self, u: int) -> np.ndarray:
        return self.main.wav(u)

    # ---- rows ----
    def spans_all(self, u: int) -> dict:
        """{table: (rows, i0, i1)} of canonical utterance ``u``."""
        return {tb: self.td[h].spans(int(self.map[h][u])) for tb, h in self.table_head.items()}

    def rows(self, name: str, utts: np.ndarray) -> np.ndarray:
        return self.td[name].rows(self.map[name][np.asarray(utts, np.int64)])

    # ---- splits ----
    def utt_split(self, fold: int) -> dict[str, np.ndarray]:
        """{'train'|'val'|'test': canonical utterance indices}; every head must agree on who is in which split (checked once per fold)."""
        if fold not in self._splits:
            ref = self.td[self.names[0]].utt_split(fold)
            for n in self.names[1:]:
                other = self.td[n].utt_split(fold)
                inv = np.full(len(self.td[n].utt_ids), -1, np.int64)
                inv[self.map[n]] = np.arange(len(self.utt_ids))
                for s in ("train", "val", "test"):
                    if set(inv[other[s]].tolist()) != set(ref[s].tolist()):
                        raise ValueError(f"{n} and {self.names[0]} disagree on the {s} utterances of fold {fold}")
            self._splits[fold] = ref
        return self._splits[fold]


def collate_mt(md: MTData, utts, device, side_t: dict, yt: dict | None, aug=None, aug_rng=None):
    """-> x, lens, spans_by_table, rows{head}, side{head}, y{head}. Pooled units of a table come out of the model in the order of ``rows``."""
    torch = FT._torch()
    wavs = [md.wav(int(u)) for u in utts]
    if aug and float(aug.get("noise_prob", 0.0)) > 0 and aug_rng is not None:          # training batches only
        wavs = [FT.add_noise(w, aug_rng, aug["snr_db"]) if aug_rng.random() < float(aug["noise_prob"]) else w for w in wavs]
    lens = np.array([len(w) for w in wavs])
    x = torch.zeros((len(wavs), int(lens.max())), dtype=torch.float32)
    for b, w in enumerate(wavs):
        x[b, :len(w)] = torch.from_numpy(w)
    sp = [md.spans_all(int(u)) for u in utts]
    spans, rows_tb, rt = {}, {}, {}
    for tb in md.table_head:
        spans[tb] = [(s[tb][1], s[tb][2]) for s in sp]
        rows_tb[tb] = np.concatenate([s[tb][0] for s in sp])
        rt[tb] = torch.as_tensor(rows_tb[tb], device=device)
    rows = {n: rows_tb[md.tasks[n].table] for n in md.names}
    side = {n: (None if side_t is None else side_t[n][rt[md.tasks[n].table]]) for n in md.names}
    ys = {n: (None if yt is None else yt[n][rt[md.tasks[n].table]]) for n in md.names}
    return x.to(device), lens, spans, rows, side, ys


# --------------------------------------------------------------------------- #
# model (torch imported lazily)
# --------------------------------------------------------------------------- #
def pool_units(H, lens, spans, kind: str):
    """H (B, K, T, D) -> (n_units, K, d_pool): identical to FineTuneModel.pool (sentence = mean||std over frames, word / phone = mean over the MFA frames)."""
    torch = FT._torch()
    out = []
    for b in range(H.shape[0]):
        Tb = se.n_frames(int(lens[b]))
        hb = H[b, :, :Tb]
        if kind == "utt":
            out.append(torch.cat([hb.mean(1), torch.sqrt(hb.var(1, correction=0) + 1e-8)], -1)[None])
        else:
            i0, i1 = spans[b]
            out.append(torch.einsum("nt,ktd->nkd", FT.pool_matrix(Tb, i0, i1, H.device), hb))
    return torch.cat(out, 0)


def build_mt_model_class():
    torch = FT._torch()
    nn, F = torch.nn, torch.nn.functional

    class MultiTaskModel(nn.Module):
        """One HF speech encoder shared by every head. Per head: [shared pooling of its table] -> LayerNorm -> learned layer-weighted sum -> [+ side] -> head."""

        def __init__(self, encoder, layer_ids: list[int], specs: list[dict], head: str, hidden: int, dropout: float, alpha_mode: str = "per_head"):
            super().__init__()
            self.enc, self.layer_ids, self.specs = encoder, list(layer_ids), list(specs)
            self.alpha_mode = alpha_mode
            D = int(encoder.config.hidden_size)
            K = len(self.layer_ids)
            self.table_kind = {s["table"]: s["pool_kind"] for s in self.specs}
            self.heads = nn.ModuleDict({head_key(s["name"]): FH.make_head((2 * D if s["pool_kind"] == "utt" else D) + s["d_side"], s["d_out"], head,
                                                                           {"hidden": hidden, "dropout": dropout}) for s in self.specs})
            keys = [head_key(s["name"]) for s in self.specs] if alpha_mode == "per_head" else ["shared"]
            self.alpha = nn.ParameterDict({k: nn.Parameter(torch.zeros(K), requires_grad=K > 1) for k in keys})
            self.train_encoder = True              # False during the head warm-up: encoder in eval mode, no graph through it (a real linear probe)

        def hidden_states(self, wav, amp: bool):
            dev_type = wav.device.type
            with torch.set_grad_enabled(torch.is_grad_enabled() and self.train_encoder), \
                    torch.autocast(device_type=dev_type if dev_type != "mps" else "cpu", dtype=torch.bfloat16, enabled=amp and dev_type == "cuda"):
                hs = self.enc(wav, output_hidden_states=True).hidden_states
            return torch.stack([hs[i] for i in self.layer_ids], 1).float()               # (B, K, T, D)

        def forward(self, wav, lens, spans, side, amp: bool = False) -> dict:
            """spans: {table: per-item (i0, i1)}; side: {head: (n, d_side) or None}. -> {head: (n_rows, d_out)}."""
            H = self.hidden_states(wav, amp)
            pooled = {}
            for tb, kind in self.table_kind.items():
                P = pool_units(H, lens, spans.get(tb), kind)
                pooled[tb] = F.layer_norm(P, (P.shape[-1],))
            out = {}
            for s in self.specs:
                k = head_key(s["name"])
                a = torch.softmax(self.alpha[k if self.alpha_mode == "per_head" else "shared"], 0)
                z = (pooled[s["table"]] * a[None, :, None]).sum(1)
                sd = side.get(s["name"]) if side else None
                if sd is not None and sd.shape[1]:
                    z = torch.cat([z, sd], 1)
                out[s["name"]] = self.heads[k](z)
            return out

        def layer_weights(self) -> dict:
            if self.alpha_mode == "per_head":
                return {s["name"]: torch.softmax(self.alpha[head_key(s["name"])], 0).detach().cpu().numpy() for s in self.specs}
            w = torch.softmax(self.alpha["shared"], 0).detach().cpu().numpy()
            return {s["name"]: w for s in self.specs}

    return MultiTaskModel


def param_groups_mt(model, tcfg: dict, extra=None) -> list[dict]:
    """AdamW groups with the same 'kind' keys as FT.param_groups ('head' | 'alpha' | 'enc'); ``extra`` = learnable loss-weight params (kind 'head')."""
    nl = int(model.enc.config.num_hidden_layers)
    dec = float(tcfg.get("layer_lr_decay", 1.0))
    head = [p for n, p in model.named_parameters() if n.startswith("heads.") and p.requires_grad]
    groups = [{"params": head, "kind": "head", "base_lr": tcfg["lr_head"], "lr": tcfg["lr_head"], "weight_decay": tcfg["wd_head"]}]
    alphas = [p for p in model.alpha.values() if p.requires_grad]
    if alphas:
        groups.append({"params": alphas, "kind": "alpha", "base_lr": tcfg["lr_alpha"], "lr": tcfg["lr_alpha"], "weight_decay": 0.0})
    if extra:
        lr = float((tcfg.get("loss") or {}).get("lr", tcfg["lr_head"]))
        groups.append({"params": list(extra), "kind": "head", "base_lr": lr, "lr": lr, "weight_decay": 0.0})
    buckets: dict[tuple, list] = {}
    for n, p in model.enc.named_parameters():
        if not p.requires_grad:
            continue
        nodecay = n.endswith(".bias") or "layer_norm" in n or "group_norm" in n or "masked_spec_embed" in n
        buckets.setdefault((FT.layer_index(n), nodecay), []).append(p)
    for (li, nodecay), ps in sorted(buckets.items()):
        lr = tcfg["lr_encoder"] * dec ** (nl - li)
        groups.append({"params": ps, "kind": "enc", "base_lr": lr, "lr": lr, "weight_decay": 0.0 if nodecay else tcfg["wd_encoder"], "layer": li})
    return groups


def probe_parameters(model) -> list:
    """Gradient probe for US 6.2: the trainable parameters of the highest transformer layer that has any (the last shared layer)."""
    best = {}
    for n, p in model.enc.named_parameters():
        li = FT.layer_index(n)
        if p.requires_grad and li >= 1:
            best.setdefault(li, []).append(p)
    return best[max(best)] if best else []


def head_gram(losses: dict, names: list[str], params: list):
    """Gram matrix (K, K) of the heads' loss gradients on ``params`` (float64 numpy). Must run before the combined loss is backpropagated."""
    torch = FT._torch()
    vecs = []
    for n in names:
        g = torch.autograd.grad(losses[n], params, retain_graph=True, allow_unused=True)
        vecs.append(torch.cat([(gi if gi is not None else torch.zeros_like(p)).reshape(-1).float() for gi, p in zip(g, params)]))
    V = torch.stack(vecs)
    return (V @ V.T).double().cpu().numpy()


# --------------------------------------------------------------------------- #
# one job = one group, one fold, one seed
# --------------------------------------------------------------------------- #
def run_group(group: str, md: MTData, enc_cache: dict, cfg: dict, tag: str, fold: int, seed: int, device: str, log=print) -> dict:
    """Train the shared-encoder model on one (group, fold, seed); write one npz per head + the group history + the checkpoint."""
    torch = FT._torch()
    nn = torch.nn
    mcfg, tcfg = cfg["model"], cfg["train"]
    names, tasks = md.names, md.tasks
    K = len(names)
    t_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    amp = tcfg.get("amp", "bf16") != "off" and dev.type == "cuda"

    # ---- splits ----
    us = md.utt_split(fold)
    tr_u, va_u, te_u = us["train"], us["val"], us["test"]
    tr_rows = {n: md.rows(n, tr_u) for n in names}

    # ---- per-head targets, losses, side features (exactly the single-task recipe) ----
    st: dict[str, dict] = {}
    for n in names:
        task, td = tasks[n], md.td[n]
        reg = task.kind == "reg"
        yi = td.yi
        d = {"task": task, "reg": reg, "K": None if reg else len(task.labels)}
        if reg:
            d["mu"], d["sd"] = float(yi[tr_rows[n]].mean()), float(yi[tr_rows[n]].std()) or 1.0
            d["yt"] = torch.as_tensor((yi - d["mu"]) / d["sd"], dtype=torch.float32, device=dev)
            d["loss_fn"] = nn.MSELoss()
        else:
            d["yt"] = torch.as_tensor(yi, dtype=torch.long, device=dev)
            cw = FH.balanced_weights(yi[tr_rows[n]], d["K"]) if tcfg.get("cw") == "balanced" else None
            d["loss_fn"] = nn.CrossEntropyLoss(weight=None if cw is None else torch.as_tensor(cw, dtype=torch.float32, device=dev))
        mu_s, sd_s = td.side[tr_rows[n]].mean(0), td.side[tr_rows[n]].std(0)
        sd_s[sd_s < 1e-6] = 1.0
        d["side_t"] = torch.as_tensor((td.side - mu_s) / sd_s, dtype=torch.float32, device=dev)
        st[n] = d
    side_t = {n: st[n]["side_t"] for n in names}
    yt = {n: st[n]["yt"] for n in names}

    # ---- model ----
    enc, _ = enc_cache["load"](mcfg)
    layers = list(range(int(mcfg["layers"][0]), int(mcfg["layers"][1]) + 1)) if mcfg["layer_mode"] == "ws" else [int(mcfg["layers"][1])]
    Model = enc_cache.setdefault("mt_cls", build_mt_model_class())
    specs = [{"name": n, "table": tasks[n].table, "pool_kind": md.td[n].pool_kind, "d_side": md.td[n].side.shape[1], "d_out": 1 if st[n]["reg"] else st[n]["K"]}
             for n in names]
    model = Model(enc, layers, specs, mcfg["head"], int(mcfg["hidden"]), float(mcfg["head_dropout"]), mcfg.get("alpha_mode", "per_head")).to(dev)
    weighter = make_weighter(tcfg.get("loss"), names, {n: tasks[n].kind for n in names}, device=dev)
    opt = torch.optim.AdamW(param_groups_mt(model, tcfg, weighter.parameters()))
    diag_on = (tcfg.get("diag") or {}).get("probe") == "top_layer"
    probe = probe_parameters(model) if (diag_on or weighter.needs_grad_norms) else []     # GradNorm needs the probe even when the US 6.2 diagnostics are off
    diag_every = int((tcfg.get("diag") or {}).get("every_steps", 0)) if (probe and diag_on) else 0
    if weighter.needs_grad_norms and not probe:
        raise ValueError(f"loss scheme {weighter.scheme!r} needs the gradient probe, but the encoder has no trainable transformer layer (check train.freeze_layers)")
    rng = np.random.default_rng(seed)
    aug, aug_rng = tcfg.get("augment") or {}, np.random.default_rng(seed + 1000)
    max_s, tol = int(tcfg["max_batch_seconds"] * SR), float(tcfg["pad_tolerance"])
    steps_per_epoch = len(FT.make_batches(md.lengths[tr_u], max_s, tol))
    total = steps_per_epoch * int(tcfg["max_epochs"])
    freeze = int(round(float(tcfg["head_warmup_epochs"]) * steps_per_epoch))
    warm = int(round(float(tcfg["warmup_frac"]) * max(total - freeze, 1)))

    def decode(n, out):
        return (out.squeeze(-1) * st[n]["sd"] + st[n]["mu"]) if st[n]["reg"] else torch.softmax(out, 1)

    def lossf(n, out, y):
        return st[n]["loss_fn"](out.squeeze(-1) if st[n]["reg"] else out, y)

    @torch.no_grad()
    def predict(utts, exact: bool) -> dict:
        model.eval()
        lens_all = md.lengths[utts]
        batches = [[i] for i in range(len(utts))] if exact else FT.make_batches(lens_all, int(tcfg["eval_batch_seconds"] * SR), tol)
        rows_l = {n: [] for n in names}
        out_l = {n: [] for n in names}
        for bidx in batches:
            x, lens, spans, rows, side, _ = collate_mt(md, utts[bidx], dev, side_t, None)
            outs = model(x, lens, spans, side, amp)
            for n in names:
                rows_l[n].append(rows[n])
                out_l[n].append(outs[n].float())
        return {n: (np.concatenate(rows_l[n]), torch.cat(out_l[n])) for n in names}

    def score_head(n, rows, out):
        """(loss, primary) of one head's outputs for ``rows`` (primary: RMSE, or -macro-F1 with argmax) - as the single-task job."""
        task = st[n]["task"]
        loss = float(lossf(n, out, yt[n][torch.as_tensor(rows, device=dev)]))
        pred = decode(n, out).cpu().numpy()
        y_raw = md.td[n].y_raw
        if st[n]["reg"]:
            return loss, FH.primary_score(task, y_raw[rows], pred)
        return loss, FH.primary_score(task, y_raw[rows], FH.decode_labels(task, pred.argmax(1)))

    probe_u = tr_u[np.random.default_rng(1).permutation(len(tr_u))[:int(tcfg["train_eval_utts"])]]
    keys = ("train_loss", "val_loss", "val_primary", "trsub_loss", "trsub_primary", "lr_enc")
    hist = {n: {k: [] for k in keys} for n in names}
    ghist = {"mean_rel": [], "gnorm": [], "gcos": [], "gn": [], "w": []}
    ref_err: dict[str, float] = {}
    head_best = {n: (np.inf, -1) for n in names}
    best, best_ep, bad, best_state = np.inf, -1, 0, None
    step = 0
    for ep in range(int(tcfg["max_epochs"])):
        model.train()
        run_loss = {n: 0.0 for n in names}
        n_units = {n: 0 for n in names}
        grams, wsum, wn = [], np.zeros(K), 0
        for bidx in FT.make_batches(md.lengths[tr_u], max_s, tol, rng):
            e_s, h_s = FT.lr_scales(step, total, freeze, warm, tcfg["schedule"])
            model.train_encoder = step >= freeze
            model.enc.train(model.train_encoder)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * (e_s if g["kind"] == "enc" else h_s)
            x, lens, spans, rows, side, ys = collate_mt(md, tr_u[bidx], dev, side_t, yt, aug, aug_rng)
            outs = model(x, lens, spans, side, amp)
            losses = {n: lossf(n, outs[n], ys[n]) for n in names}
            weighter.observe(losses)
            k_enc = step - freeze
            is_diag = bool(diag_every and model.train_encoder and k_enc % diag_every == 0)
            is_upd = bool(weighter.needs_grad_norms and probe and model.train_encoder and weighter.wants_norms(k_enc))
            G = head_gram(losses, names, probe) if (is_diag or is_upd) else None      # before any backward of the combined loss
            if is_diag:
                grams.append(G)
            if is_upd:
                weighter.update(losses, cosine_from_gram(G)[0])
            loss, w = weighter.combine(losses)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if tcfg.get("grad_clip"):
                nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], float(tcfg["grad_clip"]))
            opt.step()
            for n in names:
                run_loss[n] += float(losses[n].detach()) * len(rows[n])
                n_units[n] += len(rows[n])
            wsum += np.array([float(w[n]) for n in names])
            wn += 1
            step += 1
        # ---- end of epoch: validation per head, train-subsample probe, group criterion ----
        vals = predict(va_u, exact=False)
        trs = predict(probe_u, exact=False)
        errs, vprim = {}, {}
        for n in names:
            v_loss, v_prim = score_head(n, *vals[n])
            t_loss, t_prim = score_head(n, *trs[n])
            h = hist[n]
            h["train_loss"].append(run_loss[n] / max(n_units[n], 1)); h["val_loss"].append(v_loss); h["val_primary"].append(v_prim)
            h["trsub_loss"].append(t_loss); h["trsub_primary"].append(t_prim); h["lr_enc"].append(e_s)
            errs[n], vprim[n] = error_from_primary(st[n]["task"].kind, v_prim), v_prim
            if v_prim < head_best[n][0] - 1e-9:
                head_best[n] = (v_prim, ep)
        if ep == 0:
            ref_err = dict(errs)
        crit = mean_rel(errs, ref_err)
        ghist["mean_rel"].append(crit)
        ghist["w"].append(wsum / max(wn, 1))
        if grams:
            ns, cs = zip(*[cosine_from_gram(G) for G in grams])
            ghist["gnorm"].append(np.mean(ns, 0)); ghist["gcos"].append(np.mean(cs, 0)); ghist["gn"].append(len(grams))
        else:
            ghist["gnorm"].append(np.full(K, np.nan)); ghist["gcos"].append(np.full((K, K), np.nan)); ghist["gn"].append(0)
        log(f"    ep{ep:02d} mean_rel {crit:.4f} | " + " ".join(f"{n.split('_', 1)[-1][:9]} {vprim[n]:.3f}" for n in names)
            + f" | lr_enc {e_s:.2f}" + (f" | grad share " + " ".join(f"{s:.2f}" for s in grad_shares(ghist['gnorm'][-1])) if grams else ""))
        if best_state is None or crit < best - 1e-4:
            best, best_ep, bad = crit, ep, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(tcfg["patience"]) and step > freeze:
                break
    model.load_state_dict(best_state)
    model.train_encoder = True

    # ---- final predictions, batch size 1 (exact), one pass per split for all heads ----
    def final(utts) -> dict:
        outs = predict(utts, exact=True)
        return {n: (outs[n][0], decode(n, outs[n][1]).cpu().numpy()) for n in names}

    res: dict[str, dict] = {n: {} for n in names}
    for split, utts in (("val", va_u), ("test", te_u)):
        fo = final(utts)
        for n in names:
            rows, pred = fo[n]
            order = np.argsort(rows)
            rows, pred = rows[order], pred[order]
            res[n][f"{split}_rows"] = rows
            if st[n]["reg"]:
                res[n][f"{split}_pred"] = pred.astype(np.float32)
            else:
                res[n][f"{split}_proba"] = pred.astype(np.float32)
                res[n][f"{split}_pred"] = FH.decode_labels(st[n]["task"], pred.argmax(1))
    # train subsample per head by the single-task rule (random whole utterances until final_train_eval_rows rows); the forward pass covers the longest prefix
    perm = np.random.default_rng(0).permutation(len(tr_u))
    prefix = {}
    for n in names:
        cum = np.cumsum([len(md.td[n].rows_of[md.map[n][tr_u[i]]]) for i in perm])
        prefix[n] = int(np.searchsorted(cum, int(tcfg["final_train_eval_rows"]))) + 1
    fo = final(tr_u[perm[:max(prefix.values())]])
    train_score, val_score = {}, {}
    for n in names:
        task = st[n]["task"]
        keep = np.isin(fo[n][0], md.rows(n, tr_u[perm[:prefix[n]]]))
        rows, pred = fo[n][0][keep], fo[n][1][keep]
        order = np.argsort(rows)
        rows, pred = rows[order], pred[order]
        res[n]["train_rows"] = rows
        y_raw = md.td[n].y_raw
        if st[n]["reg"]:
            res[n]["train_pred"] = pred.astype(np.float32)
            train_score[n] = FH.primary_score(task, y_raw[rows], pred)
        else:
            res[n]["train_proba"] = pred.astype(np.float32)
            res[n]["train_y"] = md.td[n].yi[rows]
            res[n]["train_pred"] = FH.decode_labels(task, pred.argmax(1))
            train_score[n] = FH.primary_score(task, y_raw[rows], res[n]["train_pred"])
            res[n]["classes"] = np.array(task.labels).astype(str)
        val_score[n] = FH.primary_score(task, y_raw[res[n]["val_rows"]], res[n]["val_pred"])

    fit_s = time.time() - t_start
    lw = model.layer_weights()
    for n in names:
        out = job_file(cfg, tag, n, fold, seed)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, train_score=train_score[n], val_score=val_score[n], n_train=len(tr_rows[n]), best_epoch=best_ep, epochs_run=len(ghist["mean_rel"]),
                            seed=seed, fit_s=fit_s, layer_weights=lw[n], steps_per_epoch=steps_per_epoch, freeze_steps=freeze, holdout_seen=0.0,
                            mt_group=group, mt_n_heads=K, mt_group_fit_s=fit_s, mt_head_best_epoch=head_best[n][1],
                            **{f"hist_{k}": np.array(v, float) for k, v in hist[n].items()}, **res[n])
    gout = group_file(cfg, tag, group, fold, seed)
    gout.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(gout, heads=np.array(names), kinds=np.array([tasks[n].kind for n in names]), best_epoch=best_ep, fit_s=fit_s, seed=seed, fold=fold,
                        steps_per_epoch=steps_per_epoch, freeze_steps=freeze, probe_n_params=int(sum(p.numel() for p in probe)), diag_every=diag_every,
                        scheme=weighter.scheme, ref_err=np.array([ref_err[n] for n in names]),
                        hist_mean_rel=np.array(ghist["mean_rel"], float), hist_gnorm=np.array(ghist["gnorm"], float), hist_gcos=np.array(ghist["gcos"], float),
                        hist_gn=np.array(ghist["gn"], int), hist_w=np.array(ghist["w"], float),
                        head_best_epoch=np.array([head_best[n][1] for n in names]),
                        n_trainable=int(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    if tcfg.get("save_checkpoint", "best") == "best":
        ck = ckpt_file(cfg, tag, group, fold, seed)
        ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in best_state.items()}, ck)
    del model, opt
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
    return {"val": val_score, "train": train_score, "best_epoch": best_ep, "epochs": len(ghist["mean_rel"]), "fit_s": fit_s, "peak_gb": peak,
            "head_best": {n: head_best[n][1] for n in names}}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def group_tasks(cfg: dict, group: str, registry: dict[str, Task]) -> list[Task]:
    return [registry[n] for n in cfg["groups"][group]["tasks"]]


def cmd_run(args) -> None:
    cfg = load_config(args.variant)
    if args.epochs:                                                      # smoke tests only; headline runs use the config
        cfg["train"]["max_epochs"] = int(args.epochs)
    registry = build_tasks(load_baseline_config())
    device = FH.pick_device("auto")
    cache = {"device": device}
    cache["load"] = FT.make_loader(cfg, cache)
    do_norm = FT.peek_normalize(cfg)
    t0, fitted = time.time(), 0
    for group in args.groups or list(cfg["groups"]):
        tasks = group_tasks(cfg, group, registry)
        md = None
        for fold in (args.folds if args.folds is not None else tasks[0].folds):
            for seed in seeds_for_group(cfg, registry, group, args.seeds):
                done = all(job_file(cfg, args.tag, t.name, fold, seed).exists() for t in tasks) and group_file(cfg, args.tag, group, fold, seed).exists()
                if done and not args.force:
                    continue
                if args.budget and time.time() - t0 > args.budget:
                    print(f"budget of {args.budget:.0f}s used: {fitted} fitted; run again to continue")
                    return
                md = md or MTData(tasks, do_norm)
                print(f"=== {args.tag} {group} ({len(tasks)} heads) fold{fold} seed{seed} ({device})")
                r = run_group(group, md, cache, cfg, args.tag, fold, seed, device)
                fitted += 1
                print(f"  -> best_epoch {r['best_epoch']} of {r['epochs']} ({r['fit_s']:.0f}s, {r['fit_s'] / max(r['epochs'], 1):.0f}s/epoch, peak GPU {r['peak_gb']:.1f} GB)")
                for n in md.names:
                    print(f"     {n:22s} val {abs(r['val'][n]):.4f} train {abs(r['train'][n]):.4f} (head's own best epoch {r['head_best'][n]})")
    print(f"finished: {fitted} fitted")


def cmd_status(args) -> None:
    cfg = load_config(args.variant)
    registry = build_tasks(load_baseline_config())
    for group in args.groups or list(cfg["groups"]):
        tasks = group_tasks(cfg, group, registry)
        seeds = seeds_for_group(cfg, registry, group)
        total = len(tasks[0].folds) * len(seeds)
        done = sum(group_file(cfg, args.tag, group, f, s).exists() and all(job_file(cfg, args.tag, t.name, f, s).exists() for t in tasks)
                   for f in tasks[0].folds for s in seeds)
        print(f"{group:8s} {done:2d}/{total}  ({', '.join(t.name for t in tasks)})")


def cmd_dry_run(args) -> None:
    """Build the groups on the real tables: every head must label the same utterances and agree on the splits; prints sizes and steps per epoch."""
    cfg = load_config(args.variant)
    registry = build_tasks(load_baseline_config())
    tcfg = cfg["train"]
    for group in args.groups or list(cfg["groups"]):
        tasks = group_tasks(cfg, group, registry)
        md = MTData(tasks, False)
        print(f"== {group}: {len(md.utt_ids)} utterances, {len(md.names)} heads, folds {md.folds}, tables {list(md.table_head)}")
        for fold in md.folds:
            us = md.utt_split(fold)                                      # raises on any disagreement between heads
            b = FT.make_batches(md.lengths[us["train"]], int(tcfg["max_batch_seconds"] * SR), float(tcfg["pad_tolerance"]))
            print(f"   fold{fold}: train/val/test utts {len(us['train'])}/{len(us['val'])}/{len(us['test'])}  steps/epoch {len(b)}  (heads agree on all splits)")
        for n in md.names:
            td = md.td[n]
            print(f"   {n:22s} rows {len(td.meta):7d} pool {td.pool_kind:7s} side {td.side.shape[1]:3d} table {md.tasks[n].table}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "status", "dry-run"):
        p = sub.add_parser(name)
        p.add_argument("--tag", default="mt_base")
        p.add_argument("--variant", default=None)
        p.add_argument("--groups", nargs="*", choices=["so762", "l2"])
        if name == "run":
            p.add_argument("--folds", nargs="*", type=int)
            p.add_argument("--seeds", nargs="*", type=int)
            p.add_argument("--budget", type=float, default=None, help="stop starting new jobs after this many seconds")
            p.add_argument("--force", action="store_true")
            p.add_argument("--epochs", type=int, default=None, help="override train.max_epochs (smoke tests)")
    args = ap.parse_args(argv)
    {"run": cmd_run, "status": cmd_status, "dry-run": cmd_dry_run}[args.cmd](args)


if __name__ == "__main__":
    main()
