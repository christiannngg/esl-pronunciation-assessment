"""
US 5.1 - end-to-end fine-tuning of a pretrained speech encoder (WavLM / HuBERT / wav2vec 2.0) with one task head per task.

    python -m src.models.finetune dry-run                                   # build tasks, batches, spans (no torch model, no GPU)
    python -m src.models.finetune verify --n 20                             # fine-tuning pooling == cached frozen embeddings (needs real model + cache)
    python -m src.models.finetune run --tag base                            # all tasks, all folds, all seeds (resumable; skips finished jobs)
    python -m src.models.finetune run --tag base --tasks sent_accuracy --folds 0 --seeds 0 --budget 3000
    python -m src.models.finetune run --tag base__seen --holdout-seen 0.2   # US 5.2 diagnostic (seen speakers, unseen utterances)
    python -m src.models.finetune run --tag fix1 --variant fix1             # US 5.3: config overrides from configs/finetune.yaml `variants`
    python -m src.models.finetune status --tag base

What is the same as the frozen Sprint 4 heads (so the only change is encoder adaptation):
  * tasks, labels, splits, rows and their order (the Sprint 3 meta tables); SO762 = official test + 19 val speakers, L2-ARCTIC = 4 grouped folds;
  * the pooling unit and rule (src/features/ssl_embeddings.py): sentence = mean||std over all frames, word / phone = mean over the frames whose
    centre lies in the MFA interval, L2-ARCTIC phone = the FIXED window around the annotated midpoint (no annotated duration is visible);
  * side features (canonical phone one-hot, durations, ...) concatenated to the pooled vector, standardised on the training rows only;
  * MSE (regression, target standardised on train) / balanced class weights (classification); US 3.3 decision weights are applied in the report.
What differs:
  * the encoder is trained (CNN front end frozen); the layer-weighted sum standardises each layer's pooled vector with a non-affine LayerNorm
    (train-set statistics would drift while the encoder is changing);
  * whole utterances are the unit of batching (length-bucketed). Predictions for the headline numbers use batch size 1, exactly like the
    frozen extraction, because the group-norm conv front end of these checkpoints depends on padding.

One job = (tag, task, fold, seed) -> experiments/results/finetune/<tag>/<task>/fold<k>/seed<s>.npz (+ best checkpoint under results/models/finetune/).
"""

from __future__ import annotations

import argparse
import copy
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features import ssl_embeddings as se
from src.models import frozen_heads as FH
from src.models.baselines import REPO, Task, build_tasks, split_masks
from src.models.baselines import load_config as load_baseline_config

SR = se.SR
CFG_PATH = REPO / "configs" / "finetune.yaml"


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def load_config(variant: str | None = None) -> dict:
    cfg = yaml.safe_load(CFG_PATH.read_text())
    if variant:
        if variant not in cfg.get("variants", {}):
            raise KeyError(f"unknown variant {variant!r}; configs/finetune.yaml defines {sorted(cfg.get('variants', {}))}")
        cfg = deep_update(cfg, cfg["variants"][variant])
    return cfg


def deep_update(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def job_file(cfg: dict, tag: str, task: str, fold: int, seed: int) -> Path:
    return REPO / cfg["root"] / tag / task / f"fold{fold}" / f"seed{seed}.npz"


def ckpt_file(cfg: dict, tag: str, task: str, fold: int, seed: int) -> Path:
    return REPO / cfg["ckpt_root"] / tag / task / f"fold{fold}_seed{seed}.pt"


# --------------------------------------------------------------------------- #
# pure helpers (numpy only, unit-tested)
# --------------------------------------------------------------------------- #
def normalize_wave(w: np.ndarray, do_normalize: bool) -> np.ndarray:
    """Per-utterance zero-mean / unit-variance, exactly the Hugging Face feature extractor's rule (eps 1e-7) when the checkpoint asks for it."""
    w = np.asarray(w, np.float32)
    if do_normalize:
        w = (w - w.mean()) / np.sqrt(w.var() + 1e-7)
    return w.astype(np.float32)


def make_batches(lengths: np.ndarray, max_samples: int, tol: float, rng: np.random.Generator | None = None) -> list[list[int]]:
    """Length-bucketed batches of indices into ``lengths``. Every batch holds at most ``max_samples`` padded samples (batch * longest) and
    lengths within a factor (1 + tol) of the shortest, which bounds the padding. ``rng`` jitters the sort key (batches differ between epochs)
    and shuffles the batch order; ``rng=None`` is deterministic. A single utterance longer than ``max_samples`` is its own batch."""
    lengths = np.asarray(lengths, np.int64)
    key = lengths.astype(float)
    if rng is not None:
        key = key * (1.0 + rng.uniform(0.0, max(tol, 1e-6) / 2.0, len(lengths)))
    order = np.argsort(key, kind="stable")
    batches, cur = [], []
    lo = hi = 0
    for i in order:
        L = int(lengths[i])
        if cur and (max(hi, L) > min(lo, L) * (1.0 + tol) or (len(cur) + 1) * max(hi, L) > max_samples):
            batches.append(cur)
            cur = []
        if not cur:
            lo = hi = L
        lo, hi = min(lo, L), max(hi, L)
        cur.append(int(i))
    if cur:
        batches.append(cur)
    if rng is not None:
        batches = [batches[j] for j in rng.permutation(len(batches))]
    return batches


def lr_scales(step: int, total: int, freeze: int, warm: int, kind: str = "linear", head_ramp: int = 10) -> tuple[float, float]:
    """(encoder scale, head scale) in [0, 1] for optimiser step ``step``. The encoder is frozen for the first ``freeze`` steps (the head trains
    alone, linear-probe-then-fine-tune); afterwards the encoder warms up linearly over ``warm`` steps and both decay to 0 at ``total``."""
    head = min(1.0, (step + 1) / max(head_ramp, 1))
    if step < freeze:
        return 0.0, head
    s, n = step - freeze, max(total - freeze, 1)
    frac = min(max(s / n, 0.0), 1.0)
    decay = 1.0 - frac if kind == "linear" else 0.5 * (1.0 + math.cos(math.pi * frac))
    enc = decay * (min(1.0, (s + 1) / warm) if warm > 0 else 1.0)
    return enc, decay * (head if head < 1.0 else 1.0)


def holdout_utterances(speakers: np.ndarray, utt_ids: np.ndarray, frac: float, seed: int = 0) -> np.ndarray:
    """Boolean mask over utterances: ``frac`` of every speaker's utterances (at least 1 when the speaker has >= 2), chosen with a fixed seed."""
    rng = np.random.default_rng(seed)
    out = np.zeros(len(utt_ids), bool)
    for spk in pd.unique(speakers):
        idx = np.flatnonzero(speakers == spk)
        k = int(round(frac * len(idx)))
        k = max(k, 1) if len(idx) >= 2 and frac > 0 else k
        if k:
            out[rng.choice(idx, size=min(k, len(idx) - 1), replace=False)] = True
    return out


def layer_index(param_name: str) -> int:
    """Index of the hidden state a parameter feeds: transformer layer i -> i + 1; everything before the first layer -> 0."""
    m = re.search(r"encoder\.layers\.(\d+)\.", param_name)
    return int(m.group(1)) + 1 if m else 0


# --------------------------------------------------------------------------- #
# per-task data: rows, utterances, spans, side features, waveforms
# --------------------------------------------------------------------------- #
class TaskData:
    """Everything a fine-tuning job needs for one task. Rows are the Sprint 3 meta rows (same order as the frozen caches / side features)."""

    def __init__(self, task: Task, do_normalize: bool, l2_half_s: float | None = None, split_csv: Path | None = None):
        self.task = task
        self.do_normalize = do_normalize
        y, meta = FH.load_labels(task, split_csv=split_csv)
        l2_half_s = l2_half_s if l2_half_s is not None else se.load_config()["l2_window_half_s"]
        ut = se.build_unit_table(task.table, l2_half_s)
        df = ut.df
        if len(df) != len(meta) or not (df["utt_id"].to_numpy() == meta["utt_id"].to_numpy()).all() or \
                not (df["speaker"].to_numpy() == meta["speaker"].to_numpy()).all():
            raise ValueError(f"{task.table}: unit table does not match the Sprint 3 meta rows")
        self.meta, self.y_raw = meta, y
        self.yi = FH.encode_y(task, y)
        self.t0, self.t1 = df["t0"].to_numpy(float), df["t1"].to_numpy(float)
        self.pool_kind = "utt" if ut.kind == "utt" else "segment"
        S = np.load(se.tables_dir() / f"{task.table}_side_X.npy").astype(np.float32)
        if S.shape[0] != len(meta) or not np.isfinite(S).all():
            raise ValueError(f"{task.table}: side features do not match the meta rows (or are non-finite)")
        self.side = S
        codes, uniq = pd.factorize(meta["utt_id"])
        order = np.argsort(codes, kind="stable")
        bounds = np.cumsum(np.bincount(codes))[:-1]
        self.rows_of = np.split(order, bounds)                     # utterance index -> row indices (ascending)
        self.utt_of_row = np.asarray(codes, np.int64)
        self.utt_sizes = np.array([len(r) for r in self.rows_of], np.int64)
        self.utt_ids = np.asarray(uniq, dtype=object)
        first = np.array([r[0] for r in self.rows_of])
        self.utt_speaker = meta["speaker"].to_numpy()[first]
        self.dataset = task.dataset
        self._wav: dict[int, np.ndarray] = {}
        self._lengths: np.ndarray | None = None
        self._spans: dict[int, tuple] = {}

    # ---- audio ----
    def wav(self, u: int) -> np.ndarray:
        w = self._wav.get(u)
        if w is None:
            raw = se.load_wav(self.dataset, str(self.utt_ids[u]), str(self.utt_speaker[u]))
            w = self._wav[u] = normalize_wave(raw, self.do_normalize)
        return w

    @property
    def lengths(self) -> np.ndarray:
        """Samples per utterance (read from the file headers, no decoding)."""
        if self._lengths is None:
            import soundfile as sf
            self._lengths = np.array([sf.info(se.audio_path(self.dataset, str(self.utt_ids[u]), str(self.utt_speaker[u]))).frames
                                      for u in range(len(self.utt_ids))], np.int64)
        return self._lengths

    def spans(self, u: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(rows, i0, i1) of utterance ``u``: the frame range pooled for each row (sentence tasks: i0 = i1 = -1, the whole utterance)."""
        s = self._spans.get(u)
        if s is None:
            rows = self.rows_of[u]
            if self.pool_kind == "utt":
                i0 = i1 = np.full(len(rows), -1, np.int64)
            else:
                T = se.n_frames(int(self.lengths[u]))
                i0, i1 = se.unit_frames(T, self.t0[rows], self.t1[rows])
            s = self._spans[u] = (rows, i0, i1)
        return s

    # ---- splits ----
    def utt_split(self, fold: int) -> dict[str, np.ndarray]:
        """{'train'|'val'|'test': utterance indices}; every utterance lies wholly in one split (speaker-independent)."""
        m = split_masks(self.task, self.meta, fold)
        out = {}
        for name, mask in m.items():
            cnt = np.bincount(self.utt_of_row[mask], minlength=len(self.utt_ids))
            member = cnt > 0
            if not (cnt[member] == self.utt_sizes[member]).all():
                raise ValueError(f"{self.task.name}: an utterance spans two splits ({name})")
            out[name] = np.flatnonzero(member).astype(np.int64)
        if (set(out["train"]) & set(out["val"])) or (set(out["train"]) & set(out["test"])) or (set(out["val"]) & set(out["test"])):
            raise ValueError(f"{self.task.name}: splits overlap")
        return out

    def rows(self, utts: np.ndarray) -> np.ndarray:
        return np.concatenate([self.rows_of[u] for u in utts]) if len(utts) else np.zeros(0, np.int64)


# --------------------------------------------------------------------------- #
# model (torch imported lazily)
# --------------------------------------------------------------------------- #
def _torch():
    import torch
    return torch


def pool_matrix(T: int, i0, i1, device):
    """Dense (n, T) averaging matrix: row k averages frames [i0[k], i1[k])."""
    torch = _torch()
    i0, i1 = torch.as_tensor(i0, device=device), torch.as_tensor(i1, device=device)
    t = torch.arange(T, device=device)[None, :]
    inside = (t >= i0[:, None]) & (t < i1[:, None])
    return inside.float() / (i1 - i0).clamp(min=1).float()[:, None]


def build_model_class():
    torch = _torch()
    nn, F = torch.nn, torch.nn.functional

    class FineTuneModel(nn.Module):
        """HF speech encoder + (pooling, per-layer LayerNorm, learned layer-weighted sum, [side], head). One instance per task."""

        def __init__(self, encoder, layer_ids: list[int], pool_kind: str, d_side: int, d_out: int, head: str, hidden: int, dropout: float):
            super().__init__()
            self.enc, self.layer_ids, self.pool_kind = encoder, list(layer_ids), pool_kind
            self.train_encoder = True              # False during the head warm-up: encoder in eval mode, no graph through it (a real linear probe)
            D = int(encoder.config.hidden_size)
            self.d_pool = 2 * D if pool_kind == "utt" else D
            self.alpha = nn.Parameter(torch.zeros(len(self.layer_ids)), requires_grad=len(self.layer_ids) > 1)
            self.head = FH.make_head(self.d_pool + d_side, d_out, head, {"hidden": hidden, "dropout": dropout})

        def hidden_states(self, wav, amp: bool):
            dev_type = wav.device.type
            with torch.set_grad_enabled(torch.is_grad_enabled() and self.train_encoder), \
                    torch.autocast(device_type=dev_type if dev_type != "mps" else "cpu", dtype=torch.bfloat16, enabled=amp and dev_type == "cuda"):
                hs = self.enc(wav, output_hidden_states=True).hidden_states
            return torch.stack([hs[i] for i in self.layer_ids], 1).float()               # (B, K, T, D)

        def pool(self, H, lens, spans):
            """H (B, K, T, D) -> (n_units, K, d_pool). ``lens`` = samples per item; ``spans`` = per item (i0, i1) arrays (ignored for 'utt')."""
            out = []
            for b in range(H.shape[0]):
                Tb = se.n_frames(int(lens[b]))
                hb = H[b, :, :Tb]
                if self.pool_kind == "utt":
                    out.append(torch.cat([hb.mean(1), torch.sqrt(hb.var(1, correction=0) + 1e-8)], -1)[None])
                else:
                    i0, i1 = spans[b]
                    out.append(torch.einsum("nt,ktd->nkd", pool_matrix(Tb, i0, i1, H.device), hb))
            return torch.cat(out, 0)

        def forward(self, wav, lens, spans, side, amp: bool = False):
            P = self.pool(self.hidden_states(wav, amp), lens, spans)
            P = F.layer_norm(P, (P.shape[-1],))
            z = (P * torch.softmax(self.alpha, 0)[None, :, None]).sum(1)
            if side is not None and side.shape[1]:
                z = torch.cat([z, side], 1)
            return self.head(z)

        def layer_weights(self):
            return torch.softmax(self.alpha, 0).detach().cpu().numpy()

    return FineTuneModel


def load_encoder(hf_id: str, revision: str | None, mcfg: dict):
    from transformers import AutoFeatureExtractor, AutoModel
    fe = AutoFeatureExtractor.from_pretrained(hf_id, revision=revision)
    enc = AutoModel.from_pretrained(hf_id, revision=revision, **dict(mcfg.get("hf", {})))
    if mcfg.get("freeze_cnn", True):
        if hasattr(enc, "freeze_feature_encoder"):
            enc.freeze_feature_encoder()
        else:
            for p in enc.feature_extractor.parameters():
                p.requires_grad_(False)
    k = int(mcfg.get("freeze_layers", 0))
    if k > 0:                                                      # partial fine-tuning: everything below transformer layer k stays pretrained
        for mod in (enc.feature_projection, enc.encoder.pos_conv_embed, getattr(enc.encoder, "layer_norm", None)):
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad_(False)
        for layer in enc.encoder.layers[:k]:
            for p in layer.parameters():
                p.requires_grad_(False)
    return enc, bool(getattr(fe, "do_normalize", False))


def param_groups(model, tcfg: dict) -> list[dict]:
    """AdamW groups. kind: 'head' | 'alpha' | 'enc'. Encoder layers get lr * layer_lr_decay ** (12 - layer_index) (1.0 = no decay)."""
    groups, nl = [], int(model.enc.config.num_hidden_layers)
    dec = float(tcfg.get("layer_lr_decay", 1.0))
    head = [p for n, p in model.named_parameters() if n.startswith("head.") and p.requires_grad]
    groups.append({"params": head, "kind": "head", "base_lr": tcfg["lr_head"], "lr": tcfg["lr_head"], "weight_decay": tcfg["wd_head"]})
    if model.alpha.requires_grad:
        groups.append({"params": [model.alpha], "kind": "alpha", "base_lr": tcfg["lr_alpha"], "lr": tcfg["lr_alpha"], "weight_decay": 0.0})
    buckets: dict[tuple, list] = {}
    for n, p in model.enc.named_parameters():
        if not p.requires_grad:
            continue
        nodecay = n.endswith(".bias") or "layer_norm" in n or "group_norm" in n or "masked_spec_embed" in n
        buckets.setdefault((layer_index(n), nodecay), []).append(p)
    for (li, nodecay), ps in sorted(buckets.items()):
        lr = tcfg["lr_encoder"] * dec ** (nl - li)
        groups.append({"params": ps, "kind": "enc", "base_lr": lr, "lr": lr, "weight_decay": 0.0 if nodecay else tcfg["wd_encoder"], "layer": li})
    return groups


# --------------------------------------------------------------------------- #
# one job
# --------------------------------------------------------------------------- #
def collate(td: TaskData, utts, device, side_t, yt):
    torch = _torch()
    wavs = [td.wav(int(u)) for u in utts]
    lens = np.array([len(w) for w in wavs])
    x = torch.zeros((len(wavs), int(lens.max())), dtype=torch.float32)
    for b, w in enumerate(wavs):
        x[b, :len(w)] = torch.from_numpy(w)
    sp = [td.spans(int(u)) for u in utts]
    rows = np.concatenate([s[0] for s in sp])
    rt = torch.as_tensor(rows, device=device)
    return x.to(device), lens, [(s[1], s[2]) for s in sp], rows, (None if side_t is None else side_t[rt]), (None if yt is None else yt[rt])


def run_job(task: Task, td: TaskData, enc_cache: dict, cfg: dict, tag: str, fold: int, seed: int, device: str, holdout_seen: float = 0.0,
            log=print) -> dict:
    """Fine-tune on one (task, fold, seed); write predictions, history and the best checkpoint. Returns a summary dict."""
    torch = _torch()
    nn = torch.nn
    mcfg, tcfg = cfg["model"], cfg["train"]
    t_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    amp = tcfg.get("amp", "bf16") != "off" and dev.type == "cuda"
    reg = task.kind == "reg"
    K = None if reg else len(task.labels)

    # ---- splits, optional seen-speaker holdout ----
    us = td.utt_split(fold)
    tr_u, va_u, te_u = us["train"], us["val"], us["test"]
    seen_u = np.zeros(0, np.int64)
    if holdout_seen > 0:
        m = holdout_utterances(td.utt_speaker[tr_u], td.utt_ids[tr_u], holdout_seen, seed=0)
        seen_u, tr_u = tr_u[m], tr_u[~m]
    tr_rows = td.rows(tr_u)
    yi = td.yi
    if reg:
        mu_y, sd_y = float(yi[tr_rows].mean()), float(yi[tr_rows].std()) or 1.0
        yt = torch.as_tensor((yi - mu_y) / sd_y, dtype=torch.float32, device=dev)
        loss_fn = nn.MSELoss()
    else:
        yt = torch.as_tensor(yi, dtype=torch.long, device=dev)
        cw = FH.balanced_weights(yi[tr_rows], K) if tcfg.get("cw") == "balanced" else None
        loss_fn = nn.CrossEntropyLoss(weight=None if cw is None else torch.as_tensor(cw, dtype=torch.float32, device=dev))
    mu_s, sd_s = td.side[tr_rows].mean(0), td.side[tr_rows].std(0)
    sd_s[sd_s < 1e-6] = 1.0
    side_t = torch.as_tensor((td.side - mu_s) / sd_s, dtype=torch.float32, device=dev)

    # ---- model ----
    enc, _ = enc_cache["load"](mcfg)
    layers = list(range(int(mcfg["layers"][0]), int(mcfg["layers"][1]) + 1)) if mcfg["layer_mode"] == "ws" else [int(mcfg["layers"][1])]
    Model = enc_cache.setdefault("cls", build_model_class())
    model = Model(enc, layers, td.pool_kind, td.side.shape[1], 1 if reg else K, mcfg["head"], int(mcfg["hidden"]), float(mcfg["head_dropout"])).to(dev)
    groups = param_groups(model, tcfg)
    opt = torch.optim.AdamW(groups)
    rng = np.random.default_rng(seed)
    max_s, tol = int(tcfg["max_batch_seconds"] * SR), float(tcfg["pad_tolerance"])
    steps_per_epoch = len(make_batches(td.lengths[tr_u], max_s, tol))
    total = steps_per_epoch * int(tcfg["max_epochs"])
    freeze = int(round(float(tcfg["head_warmup_epochs"]) * steps_per_epoch))
    warm = int(round(float(tcfg["warmup_frac"]) * max(total - freeze, 1)))

    def decode(out):                                   # model outputs -> original-scale predictions / probabilities
        return (out.squeeze(-1) * sd_y + mu_y) if reg else torch.softmax(out, 1)

    @torch.no_grad()
    def predict(utts, exact: bool):
        model.eval()
        lens_all = td.lengths[utts]
        batches = [[i] for i in range(len(utts))] if exact else make_batches(lens_all, int(tcfg["eval_batch_seconds"] * SR), tol)
        rows_l, out_l = [], []
        for bidx in batches:
            x, lens, spans, rows, side, _ = collate(td, utts[bidx], dev, side_t, None)
            rows_l.append(rows)
            out_l.append(model(x, lens, spans, side, amp).float())
        rows, out = np.concatenate(rows_l), torch.cat(out_l)
        return rows, out

    def score_rows(rows, out):
        """(loss, primary) of model outputs ``out`` for ``rows`` (primary: RMSE, or -macro-F1 with argmax)."""
        loss = float(loss_fn(out.squeeze(-1) if reg else out, yt[torch.as_tensor(rows, device=dev)]))
        pred = decode(out).cpu().numpy()
        if reg:
            return loss, FH.primary_score(task, td.y_raw[rows], pred)
        return loss, FH.primary_score(task, td.y_raw[rows], FH.decode_labels(task, pred.argmax(1)))

    probe_u = tr_u[np.random.default_rng(1).permutation(len(tr_u))[:int(tcfg["train_eval_utts"])]]
    hist = {k: [] for k in ("train_loss", "val_loss", "val_primary", "trsub_loss", "trsub_primary", "seen_loss", "seen_primary", "lr_enc")}
    best, best_ep, bad, best_state = np.inf, -1, 0, None
    step = 0
    for ep in range(int(tcfg["max_epochs"])):
        model.train()
        run_loss, n_units = 0.0, 0
        for bidx in make_batches(td.lengths[tr_u], max_s, tol, rng):
            e_s, h_s = lr_scales(step, total, freeze, warm, tcfg["schedule"])
            model.train_encoder = step >= freeze
            model.enc.train(model.train_encoder)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * (e_s if g["kind"] == "enc" else h_s)
            x, lens, spans, rows, side, y = collate(td, tr_u[bidx], dev, side_t, yt)
            out = model(x, lens, spans, side, amp)
            loss = loss_fn(out.squeeze(-1) if reg else out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if tcfg.get("grad_clip"):
                nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], float(tcfg["grad_clip"]))
            opt.step()
            run_loss += float(loss.detach()) * len(rows)
            n_units += len(rows)
            step += 1
        vr, vo = predict(va_u, exact=False)
        v_loss, v_prim = score_rows(vr, vo)
        tr_r, tr_o = predict(probe_u, exact=False)
        t_loss, t_prim = score_rows(tr_r, tr_o)
        hist["train_loss"].append(run_loss / max(n_units, 1)); hist["val_loss"].append(v_loss); hist["val_primary"].append(v_prim)
        hist["trsub_loss"].append(t_loss); hist["trsub_primary"].append(t_prim); hist["lr_enc"].append(e_s)
        if len(seen_u):
            sr, so = predict(seen_u, exact=False)
            s_loss, s_prim = score_rows(sr, so)
        else:
            s_loss = s_prim = float("nan")
        hist["seen_loss"].append(s_loss); hist["seen_primary"].append(s_prim)
        log(f"    ep{ep:02d} train {hist['train_loss'][-1]:.4f} | val loss {v_loss:.4f} prim {v_prim:.4f} | trsub prim {t_prim:.4f}"
            + (f" | seen prim {s_prim:.4f}" if len(seen_u) else "") + f" | lr_enc {e_s:.2f}")
        if best_state is None or v_prim < best - 1e-4:
            best, best_ep, bad = v_prim, ep, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(tcfg["patience"]) and step > freeze:
                break
    model.load_state_dict(best_state)
    model.train_encoder = True

    # ---- final predictions, batch size 1 (exact) ----
    def final(utts):
        rows, out = predict(utts, exact=True)
        pred = decode(out).cpu().numpy()
        return rows, pred

    res: dict = {}
    for name, utts in (("val", va_u), ("test", te_u), ("seen", seen_u)):
        if not len(utts):
            continue
        rows, pred = final(utts)
        order = np.argsort(rows)
        rows, pred = rows[order], pred[order]
        res[f"{name}_rows"] = rows
        if reg:
            res[f"{name}_pred"] = pred.astype(np.float32)
        else:
            res[f"{name}_proba"] = pred.astype(np.float32)
            res[f"{name}_pred"] = FH.decode_labels(task, pred.argmax(1))
    # train subsample (whole utterances until final_train_eval_rows rows) scored with the best checkpoint
    perm = np.random.default_rng(0).permutation(len(tr_u))
    cum = np.cumsum([len(td.rows_of[tr_u[i]]) for i in perm])
    sub_u = tr_u[perm[: int(np.searchsorted(cum, int(tcfg["final_train_eval_rows"]))) + 1]]
    rows, pred = final(sub_u)
    order = np.argsort(rows)
    rows, pred = rows[order], pred[order]
    res["train_rows"] = rows
    if reg:
        res["train_pred"] = pred.astype(np.float32)
        train_score = FH.primary_score(task, td.y_raw[rows], pred)
    else:
        res["train_proba"] = pred.astype(np.float32)
        res["train_y"] = yi[rows]
        res["train_pred"] = FH.decode_labels(task, pred.argmax(1))
        train_score = FH.primary_score(task, td.y_raw[rows], res["train_pred"])
        res["classes"] = np.array(task.labels).astype(str)
    val_rows = res["val_rows"]
    val_score = FH.primary_score(task, td.y_raw[val_rows], res["val_pred"])
    out = job_file(cfg, tag, task.name, fold, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, train_score=train_score, val_score=val_score, n_train=len(tr_rows), best_epoch=best_ep, epochs_run=len(hist["val_loss"]),
                        seed=seed, fit_s=time.time() - t_start, layer_weights=model.layer_weights(), steps_per_epoch=steps_per_epoch,
                        freeze_steps=freeze, holdout_seen=holdout_seen, **{f"hist_{k}": np.array(v, float) for k, v in hist.items()}, **res)
    if tcfg.get("save_checkpoint", "best") == "best":
        ck = ckpt_file(cfg, tag, task.name, fold, seed)
        ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in best_state.items()}, ck)
    del model, opt
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
    return {"val": val_score, "train": train_score, "best_epoch": best_ep, "epochs": len(hist["val_loss"]), "fit_s": time.time() - t_start, "peak_gb": peak}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def seeds_for(task: Task, cfg: dict, override: list[int] | None) -> list[int]:
    return override if override else list(cfg["seeds"] if task.dataset == "so762" else cfg["l2_seeds"])


def make_loader(cfg: dict, cache: dict):
    frozen = yaml.safe_load((REPO / "configs" / "frozen.yaml").read_text())
    spec = frozen["encoders"][cfg["encoder"]]

    def load(mcfg):
        enc, norm = load_encoder(spec["hf_id"], spec.get("revision"), mcfg)
        cache["do_normalize"] = norm
        return enc.to(cache["device"]), norm
    return load


def peek_normalize(cfg: dict) -> bool:
    from transformers import AutoFeatureExtractor
    spec = yaml.safe_load((REPO / "configs" / "frozen.yaml").read_text())["encoders"][cfg["encoder"]]
    return bool(getattr(AutoFeatureExtractor.from_pretrained(spec["hf_id"], revision=spec.get("revision")), "do_normalize", False))


def cmd_run(args) -> None:
    cfg = load_config(args.variant)
    if args.epochs:                                                      # smoke tests only; headline runs use the config
        cfg["train"]["max_epochs"] = int(args.epochs)
    tasks = build_tasks(load_baseline_config())
    names = args.tasks or cfg["tasks"]
    device = FH.pick_device("auto")
    cache = {"device": device}
    cache["load"] = make_loader(cfg, cache)
    do_norm = peek_normalize(cfg)
    hs = args.holdout_seen if args.holdout_seen is not None else float(cfg.get("holdout_seen_frac", 0.0))
    t0, fitted = time.time(), 0
    for name in names:
        task = tasks[name]
        td = None
        for fold in (args.folds if args.folds is not None else task.folds):
            for seed in seeds_for(task, cfg, args.seeds):
                f = job_file(cfg, args.tag, name, fold, seed)
                if f.exists() and not args.force:
                    continue
                if args.budget and time.time() - t0 > args.budget:
                    print(f"budget of {args.budget:.0f}s used: {fitted} fitted; run again to continue")
                    return
                td = td or TaskData(task, do_norm)
                print(f"=== {args.tag} {name} fold{fold} seed{seed} ({device})")
                r = run_job(task, td, cache, cfg, args.tag, fold, seed, device, hs)
                fitted += 1
                print(f"  -> val {r['val']:.4f} train {r['train']:.4f} best_epoch {r['best_epoch']} of {r['epochs']} ({r['fit_s']:.0f}s, {r['fit_s'] / max(r['epochs'], 1):.0f}s/epoch, peak GPU {r['peak_gb']:.1f} GB)")
    print(f"finished: {fitted} fitted")


def cmd_status(args) -> None:
    cfg = load_config(args.variant)
    tasks = build_tasks(load_baseline_config())
    for name in cfg["tasks"]:
        task = tasks[name]
        seeds = seeds_for(task, cfg, None)
        done = sum(job_file(cfg, args.tag, name, f, s).exists() for f in task.folds for s in seeds)
        print(f"{name:22s} {done:2d}/{len(task.folds) * len(seeds)}")


def cmd_dry_run(args) -> None:
    cfg = load_config(args.variant)
    tasks = build_tasks(load_baseline_config())
    tcfg = cfg["train"]
    for name in args.tasks or cfg["tasks"]:
        task = tasks[name]
        td = TaskData(task, False)
        us = td.utt_split(task.folds[0])
        b = make_batches(td.lengths[us["train"]], int(tcfg["max_batch_seconds"] * SR), float(tcfg["pad_tolerance"]))
        pad = np.mean([1 - td.lengths[us["train"]][x].mean() / td.lengths[us["train"]][x].max() for x in b])
        print(f"{name:22s} rows {len(td.meta):7d} utts {len(td.utt_ids):5d} pool {td.pool_kind:7s} side {td.side.shape[1]:3d} "
              f"train/val/test utts {len(us['train'])}/{len(us['val'])}/{len(us['test'])} steps/epoch {len(b)} mean pad {pad:.3f}")


def cmd_verify(args) -> None:
    """The fine-tuning code path (frozen, eval, fp32, batch 1) must reproduce the cached Sprint 4 embeddings of the same rows."""
    torch = _torch()
    cfg = load_config(args.variant)
    tasks = build_tasks(load_baseline_config())
    device = FH.pick_device("auto")
    cache = {"device": device}
    cache["load"] = make_loader(cfg, cache)
    do_norm = peek_normalize(cfg)
    root = se.cache_root(yaml.safe_load((REPO / "configs" / "frozen.yaml").read_text()), args.cache_root) / cfg["encoder"]
    enc, _ = cache["load"]({"hf": {"layerdrop": 0.0, "apply_spec_augment": False}, "freeze_cnn": True})
    enc.eval()
    Model = build_model_class()
    worst = 0.0
    for name in args.tasks or ["sent_accuracy", "word_accuracy", "phone_score", "l2_phone_error", "l2_word_error"]:
        task = tasks[name]
        td = TaskData(task, do_norm)
        E = np.load(root / f"{task.table}_X.npy", mmap_mode="r")
        model = Model(enc, list(range(13)), td.pool_kind, 0, 1, "linear", 8, 0.0).to(device).eval()
        utts = np.random.default_rng(0).choice(len(td.utt_ids), min(args.n, len(td.utt_ids)), replace=False)
        errs = []
        for u in utts:
            rows, i0, i1 = td.spans(int(u))
            w = torch.as_tensor(td.wav(int(u))[None], device=device)
            with torch.no_grad():
                P = model.pool(model.hidden_states(w, False), [w.shape[1]], [(i0, i1)]).cpu().numpy()          # (n, 13, d_pool)
            ref = np.asarray(E[rows], np.float32)
            errs.append(np.abs(P - ref).max() / (np.abs(ref).max() + 1e-6))
        worst = max(worst, max(errs))
        print(f"{name:18s} {len(utts)} utts: max relative error vs cache {max(errs):.2e} (fp16 cache => ~1e-3 expected)")
    print("OK" if worst < 5e-3 else f"MISMATCH (worst {worst:.2e})")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "status", "dry-run", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--tag", default="base")
        p.add_argument("--variant", default=None)
        p.add_argument("--tasks", nargs="*")
        if name == "run":
            p.add_argument("--folds", nargs="*", type=int)
            p.add_argument("--seeds", nargs="*", type=int)
            p.add_argument("--holdout-seen", type=float, default=None)
            p.add_argument("--budget", type=float, default=None, help="stop starting new jobs after this many seconds")
            p.add_argument("--force", action="store_true")
            p.add_argument("--epochs", type=int, default=None, help="override train.max_epochs (smoke tests)")
        if name == "verify":
            p.add_argument("--n", type=int, default=20)
            p.add_argument("--cache-root", default=None)
    args = ap.parse_args(argv)
    {"run": cmd_run, "status": cmd_status, "dry-run": cmd_dry_run, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    main()
