"""
US 4.1 - frozen self-supervised speech-encoder embeddings, pooled to the Sprint 3 units.

    python -m src.features.ssl_embeddings --dry-run                      # build + validate the unit tables only (no torch, no GPU)
    python -m src.features.ssl_embeddings --encoder wav2vec2_base --limit 20   # smoke test into <encoder>__smoke20/
    python -m src.features.ssl_embeddings --encoder wav2vec2_base        # full extraction (resumable)
    python -m src.features.ssl_embeddings --encoder all
    python -m src.features.ssl_embeddings --encoder wav2vec2_base --verify     # check a finished cache

One forward pass per utterance (batch of 1, so no padding or attention-mask questions), ALL hidden layers kept
(index 0 = encoder input after the conv front end, 1..N = transformer layers). Frame vectors are pooled straight
to the units the Sprint 3 baselines used, in the SAME row order as ``data/processed/features/tables/<unit>_meta.csv.gz``,
so a row of ``<unit>_X.npy`` here is the same sample as that row in the MFCC table (paired bootstrap works unchanged).

Cache layout (``<cache_root>/<encoder>/``): ``<unit>_X.npy`` float16 (n_rows, n_layers, width), ``manifest.json``
(model id + resolved commit, versions, counts, key hashes), ``progress.json`` (finished shards, for resume).

Pooling rules
  * The encoder emits one vector per 20 ms (stride 320 samples, receptive field 400 samples), so frame i is centred
    at ``0.0125 + 0.02 i`` seconds. A unit [t0, t1) pools the frames whose centre lies in [t0, t1); a unit that contains
    no centre (phones shorter than ~20 ms) takes the single frame nearest its midpoint.
  * SO762 word / phone: the MFA interval (label-blind forced alignment of the canonical pronunciation).
  * L2-ARCTIC phone: the same FIXED window (+-l2arctic_window_half_s around the annotated midpoint) the baseline used, so
    the annotated duration (which would leak deletions / additions) is not visible. L2-ARCTIC word: span of the non-addition units.
  * Sentence: mean and (population) std over every frame of the file; silence is not trimmed, as in Sprint 2/3.
Pooling is done in float64 on the CPU with a cumulative sum, so it is exact up to the fp16 cast on write.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import get_data_root

REPO = Path(__file__).resolve().parents[2]
CODE_VERSION = "ssl_embeddings-v1"

# wav2vec 2.0 / HuBERT / WavLM base conv front end (same for the three base checkpoints; read from the model config when loaded)
SR = 16000
CONV_KERNEL = (10, 3, 3, 3, 3, 2, 2)
CONV_STRIDE = (5, 2, 2, 2, 2, 2, 2)
STRIDE_S = 320 / SR              # 20 ms between frames
FIRST_CENTER_S = 200 / SR        # frame 0 covers samples [0, 400) -> centre at 12.5 ms

UNITS = ("so762_sentence", "so762_word", "so762_phone", "l2_word", "l2_phone")
UNIT_KEYS = {
    "so762_sentence": ["utt_id"],
    "so762_word": ["utt_id", "word_idx"],
    "so762_phone": ["utt_id", "word_idx", "phone_idx"],
    "l2_word": ["utt_id", "word_idx"],
    "l2_phone": ["utt_id", "phone_idx"],
}
SLOT_CLASSES = ("correct", "substitution", "deletion")      # L2-ARCTIC rows that occupy a canonical slot (additions / silence do not)


# --------------------------------------------------------------------------- #
# configuration and paths
# --------------------------------------------------------------------------- #
def load_config() -> dict:
    import yaml
    cfg = yaml.safe_load((REPO / "configs" / "frozen.yaml").read_text())
    base = yaml.safe_load((REPO / "configs" / "baseline.yaml").read_text())
    cfg["l2_window_half_s"] = float(base["l2arctic_window_half_s"])       # one source of truth for the fixed window
    return cfg


def tables_dir() -> Path:
    return get_data_root() / "processed" / "features" / "tables"


def alignments_dir() -> Path:
    return get_data_root() / "processed" / "alignments"


def cache_root(cfg: dict | None = None, override: str | None = None) -> Path:
    if override:
        return Path(override)
    if os.environ.get("SSL_CACHE_ROOT"):
        return Path(os.environ["SSL_CACHE_ROOT"])
    if cfg and cfg.get("extraction", {}).get("cache_root"):
        return Path(cfg["extraction"]["cache_root"])
    return get_data_root() / "processed" / "features" / "ssl"


def audio_path(dataset: str, utt_id: str, speaker: str | None = None) -> Path:
    base = get_data_root() / "processed" / "audio" / dataset
    return base / f"{utt_id}.wav" if dataset == "so762" else base / str(speaker) / f"{utt_id}.wav"


def load_wav(dataset: str, utt_id: str, speaker: str | None = None) -> np.ndarray:
    import soundfile as sf
    y, sr = sf.read(audio_path(dataset, utt_id, speaker), dtype="float32")
    if sr != SR:
        raise ValueError(f"{utt_id}: expected {SR} Hz audio, got {sr}")
    return y.mean(1) if y.ndim > 1 else y


# --------------------------------------------------------------------------- #
# frame geometry and pooling (pure numpy, unit-tested)
# --------------------------------------------------------------------------- #
def n_frames(n_samples: int, kernel=CONV_KERNEL, stride=CONV_STRIDE) -> int:
    """Number of encoder frames for a waveform of ``n_samples`` (conv front end, no padding)."""
    n = int(n_samples)
    for k, s in zip(kernel, stride):
        if n < k:
            return 0
        n = (n - k) // s + 1
    return n


def frame_centers(T: int) -> np.ndarray:
    return FIRST_CENTER_S + STRIDE_S * np.arange(T)


def unit_frames(T: int, t0, t1) -> tuple[np.ndarray, np.ndarray]:
    """Frame index ranges [i0, i1) holding the frames whose centre lies in [t0, t1); always >= 1 frame, clamped to [0, T]."""
    if T < 1:
        raise ValueError("encoder produced no frames")
    t0, t1 = np.atleast_1d(np.asarray(t0, float)), np.atleast_1d(np.asarray(t1, float))
    i0 = np.ceil(np.round((t0 - FIRST_CENTER_S) / STRIDE_S, 6)).astype(np.int64)
    i1 = np.ceil(np.round((t1 - FIRST_CENTER_S) / STRIDE_S, 6)).astype(np.int64)
    i0, i1 = np.clip(i0, 0, T), np.clip(i1, 0, T)
    empty = i1 <= i0
    if empty.any():
        mid = 0.5 * (t0 + t1)
        j = np.clip(np.rint((mid - FIRST_CENTER_S) / STRIDE_S).astype(np.int64), 0, T - 1)
        i0 = np.where(empty, j, i0)
        i1 = np.where(empty, j + 1, i1)
    return i0, i1


def pool_units(H: np.ndarray, i0: np.ndarray, i1: np.ndarray) -> np.ndarray:
    """H (L, T, D) -> (n, L, D): mean of frames [i0, i1) per unit and layer, float64 accumulation."""
    L, T, D = H.shape
    cs = np.zeros((L, T + 1, D), np.float64)
    np.cumsum(H, axis=1, dtype=np.float64, out=cs[:, 1:, :])
    sums = cs[:, i1, :] - cs[:, i0, :]                                  # (L, n, D)
    return (sums / (i1 - i0).astype(np.float64)[None, :, None]).transpose(1, 0, 2)


def pool_utterance(H: np.ndarray) -> np.ndarray:
    """H (L, T, D) -> (L, 2D): [mean over frames, population std over frames]."""
    H64 = H.astype(np.float64, copy=False)
    return np.concatenate([H64.mean(1), H64.std(1)], axis=-1)


# --------------------------------------------------------------------------- #
# unit tables: Sprint 3 row order + the time span to pool for every row
# --------------------------------------------------------------------------- #
@dataclass
class UnitTable:
    name: str
    dataset: str                  # "so762" | "l2arctic"
    df: pd.DataFrame              # one row per Sprint 3 meta row: utt_id, speaker, key columns, t0, t1
    keys_sha256: str

    @property
    def kind(self) -> str:
        return "utt" if self.name.endswith("_sentence") else "segment"

    def width(self, dim: int) -> int:
        return 2 * dim if self.kind == "utt" else dim


def _read_meta(name: str) -> pd.DataFrame:
    return pd.read_csv(tables_dir() / f"{name}_meta.csv.gz", dtype={"utt_id": str, "speaker": str})


def _keys_hash(df: pd.DataFrame, keys: list[str]) -> str:
    h = hashlib.sha256()
    for row in df[keys].itertuples(index=False, name=None):
        h.update(("|".join(str(v) for v in row) + "\n").encode())
    return h.hexdigest()


def _attach_times(meta: pd.DataFrame, times: pd.DataFrame, keys: list[str], name: str) -> pd.DataFrame:
    """Add t0/t1 to the Sprint 3 meta rows by key, preserving meta order; every key must resolve exactly once."""
    if times.duplicated(keys).any():
        raise ValueError(f"{name}: duplicate alignment keys {keys}")
    out = meta[["utt_id", "speaker", *keys[1:]]].merge(times[[*keys, "t0", "t1"]], on=keys, how="left", validate="one_to_one")
    if len(out) != len(meta):
        raise ValueError(f"{name}: row count changed in join ({len(meta)} -> {len(out)})")
    if out[["t0", "t1"]].isna().any().any():
        raise ValueError(f"{name}: {int(out['t0'].isna().sum())} meta rows have no alignment interval")
    if not (out["t1"] > out["t0"]).all():
        raise ValueError(f"{name}: non-positive pooling span")
    return out


def build_unit_table(name: str, l2_half_s: float) -> UnitTable:
    if name not in UNITS:
        raise ValueError(f"unknown unit {name!r}; expected one of {UNITS}")
    meta = _read_meta(name)
    keys = UNIT_KEYS[name]
    ali = alignments_dir()
    if name == "so762_sentence":
        df = meta[["utt_id", "speaker"]].copy()
        df["t0"], df["t1"] = np.nan, np.nan                               # whole file
    elif name in ("so762_word", "so762_phone"):
        src = "so762_words.csv.gz" if name == "so762_word" else "so762_phones.csv.gz"
        t = pd.read_csv(ali / src, dtype={"utt_id": str}, usecols=[*keys, "start", "end"]).rename(columns={"start": "t0", "end": "t1"})
        df = _attach_times(meta, t, keys, name)
    else:
        clean = pd.read_csv(ali / "l2arctic_phones_clean.csv.gz", dtype={"utt_id": str, "speaker": str})
        clean = clean[clean["corpus"] == "arctic"]
        if name == "l2_phone":
            t = clean[["utt_id", "phone_idx", "start", "end"]].copy()
            mid = 0.5 * (t["start"] + t["end"])
            t["t0"], t["t1"] = mid - l2_half_s, mid + l2_half_s            # fixed window, as in aggregate.build_l2arctic
        else:
            slot = clean[clean["error_class"].isin(SLOT_CLASSES) & clean["word_idx"].notna()]
            g = slot.groupby(["utt_id", slot["word_idx"].astype(int)])
            t = g.agg(t0=("start", "min"), t1=("end", "max")).reset_index()
        df = _attach_times(meta, t, keys, name)
    dataset = "so762" if name.startswith("so762") else "l2arctic"
    return UnitTable(name, dataset, df.reset_index(drop=True), _keys_hash(meta, keys))


def build_tables(units, l2_half_s: float) -> dict[str, UnitTable]:
    return {u: build_unit_table(u, l2_half_s) for u in units}


def utterance_list(tables: dict[str, UnitTable], limit: int | None = None) -> pd.DataFrame:
    """dataset, utt_id, speaker - every utterance any unit needs, in first-appearance order (deterministic)."""
    parts = [t.df[["utt_id", "speaker"]].assign(dataset=t.dataset) for t in tables.values()]
    u = pd.concat(parts, ignore_index=True).drop_duplicates(["dataset", "utt_id"])[["dataset", "utt_id", "speaker"]]
    if limit:
        u = u.groupby("dataset", sort=False).head(limit)
    return u.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# frozen encoder (torch / transformers imported lazily)
# --------------------------------------------------------------------------- #
class FrozenEncoder:
    """wav (float32, 16 kHz mono) -> hidden states (n_layers, T, dim) float32. Weights are never updated."""

    def __init__(self, hf_id: str, revision: str | None = None, device: str = "auto"):
        import torch
        from transformers import AutoFeatureExtractor, AutoModel
        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.fe = AutoFeatureExtractor.from_pretrained(hf_id, revision=revision)
        if int(self.fe.sampling_rate) != SR:
            raise ValueError(f"{hf_id} expects {self.fe.sampling_rate} Hz")
        self.model = AutoModel.from_pretrained(hf_id, revision=revision).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        c = self.model.config
        self.hf_id, self.revision = hf_id, revision
        self.commit = getattr(c, "_commit_hash", None)
        self.n_layers, self.dim = int(c.num_hidden_layers) + 1, int(c.hidden_size)
        self.kernel, self.stride = tuple(c.conv_kernel), tuple(c.conv_stride)
        self.normalizes = bool(getattr(self.fe, "do_normalize", False))

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = self.fe(wav, sampling_rate=SR, return_tensors="pt").input_values.to(self.device)       # (1, N), per-checkpoint normalisation
        with torch.inference_mode():
            out = self.model(x, output_hidden_states=True)
        if len(out.hidden_states) != self.n_layers:
            raise RuntimeError(f"expected {self.n_layers} hidden states, got {len(out.hidden_states)}")
        return torch.stack(out.hidden_states, 0)[:, 0].float().cpu().numpy()                        # (L, T, D)


# --------------------------------------------------------------------------- #
# extraction core (encoder and audio loader are injected, so it is testable without torch or real data)
# --------------------------------------------------------------------------- #
def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def run_extraction(tables: dict[str, UnitTable], utts: pd.DataFrame, encode, wav_loader, out_dir: Path, *, n_layers: int, dim: int,
                   shard_size: int = 250, cache_dtype: str = "float16", fingerprint: dict | None = None,
                   conv=(CONV_KERNEL, CONV_STRIDE), progress: bool = False, stop_after_shards: int | None = None) -> dict:
    """Fill ``<out_dir>/<unit>_X.npy`` for every unit; resumes from ``progress.json``. Returns run statistics.

    ``stop_after_shards`` exists for the resume test (simulates a disconnect)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = json.loads(json.dumps(fingerprint or {}))
    state_path = out_dir / "progress.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"fingerprint": fp, "done_shards": []}
    if state["fingerprint"] != fp:
        raise RuntimeError(f"{out_dir} was built with different settings; rerun with --overwrite to rebuild it.\n"
                           f"  existing: {state['fingerprint']}\n  requested: {fp}")
    done = set(state["done_shards"])

    mms = {}
    for name, t in tables.items():
        shape = (len(t.df), n_layers, t.width(dim))
        path = out_dir / f"{name}_X.npy"
        if path.exists():
            mm = np.lib.format.open_memmap(path, mode="r+")
            if mm.shape != shape or mm.dtype != np.dtype(cache_dtype):
                raise RuntimeError(f"{path}: existing {mm.shape} {mm.dtype} != requested {shape} {cache_dtype}; use --overwrite")
        else:
            mm = np.lib.format.open_memmap(path, mode="w+", dtype=cache_dtype, shape=shape)
        mms[name] = mm
    groups = {n: t.df.groupby("utt_id", sort=False).indices for n, t in tables.items()}
    t0s = {n: t.df["t0"].to_numpy() for n, t in tables.items()}
    t1s = {n: t.df["t1"].to_numpy() for n, t in tables.items()}

    n_shards = -(-len(utts) // shard_size)
    stats = {"utterances": 0, "max_abs": {n: 0.0 for n in tables}, "frames": 0}
    it = range(n_shards)
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(it, desc="shards", unit="shard")
    ran = 0
    for sid in it:
        if sid in done:
            continue
        if stop_after_shards is not None and ran >= stop_after_shards:
            break
        for r in utts.iloc[sid * shard_size:(sid + 1) * shard_size].itertuples(index=False):
            wav = wav_loader(r.dataset, r.utt_id, r.speaker)
            H = encode(wav)
            if H.ndim != 3 or H.shape[0] != n_layers or H.shape[2] != dim:
                raise ValueError(f"{r.utt_id}: unexpected hidden-state shape {H.shape}")
            if H.shape[1] != n_frames(len(wav), *conv):
                raise ValueError(f"{r.utt_id}: {H.shape[1]} frames, expected {n_frames(len(wav), *conv)} for {len(wav)} samples")
            for name, t in tables.items():
                if t.dataset != r.dataset:
                    continue
                idx = groups[name].get(r.utt_id)
                if idx is None:
                    continue
                if t.kind == "utt":
                    v = pool_utterance(H)[None]
                else:
                    i0, i1 = unit_frames(H.shape[1], t0s[name][idx], t1s[name][idx])
                    v = pool_units(H, i0, i1)
                v16 = v.astype(cache_dtype)
                if not np.isfinite(v16).all():
                    raise FloatingPointError(f"{r.utt_id}/{name}: non-finite pooled values after the {cache_dtype} cast")
                stats["max_abs"][name] = max(stats["max_abs"][name], float(np.abs(v16).max()))
                mms[name][idx] = v16
            stats["utterances"] += 1
            stats["frames"] += H.shape[1]
        for mm in mms.values():
            mm.flush()
        done.add(sid)
        ran += 1
        _atomic_write(state_path, json.dumps({"fingerprint": fp, "done_shards": sorted(done)}))
    stats["shards_done"], stats["shards_total"] = len(done), n_shards
    stats["complete"] = len(done) == n_shards
    return stats


def verify_cache(out_dir: Path, tables: dict[str, UnitTable], dim: int, n_layers: int, utts: pd.DataFrame | None = None,
                 chunk: int = 4000) -> dict:
    """Shape, finiteness and unfilled-row check (an unfilled row is all zeros). With ``utts`` (a smoke run) only the rows of
    those utterances are expected to be filled."""
    rep = {}
    for name, t in tables.items():
        mm = np.load(out_dir / f"{name}_X.npy", mmap_mode="r")
        shape = (len(t.df), n_layers, t.width(dim))
        expected = np.ones(len(t.df), bool)
        if utts is not None:
            expected = t.df["utt_id"].isin(set(utts.loc[utts["dataset"] == t.dataset, "utt_id"])).to_numpy()
        bad = zero = 0
        for a in range(0, len(mm), chunk):
            x = np.asarray(mm[a:a + chunk])
            bad += int((~np.isfinite(x)).any(axis=(1, 2)).sum())
            zero += int(((x == 0).all(axis=(1, 2)) & expected[a:a + chunk]).sum())
        rep[name] = {"rows": int(len(mm)), "expected_filled": int(expected.sum()), "shape_ok": tuple(mm.shape) == shape,
                     "non_finite_rows": bad, "unfilled_rows": zero}
    return rep


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _describe(tables: dict[str, UnitTable], dim: int = 768, layers: int = 13) -> None:
    total = 0
    for n, t in tables.items():
        b = len(t.df) * layers * t.width(dim) * 2
        total += b
        span = "whole file" if t.kind == "utt" else f"median span {np.median(t.df['t1'] - t.df['t0']):.3f} s"
        print(f"  {n:15s} rows={len(t.df):7d}  {span:22s} ~{b / 1e9:.2f} GB (fp16, {layers}x{dim})  keys sha256={t.keys_sha256[:12]}")
    print(f"  total ~{total / 1e9:.1f} GB per base encoder")


def main(argv=None):
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", choices=[*cfg["encoders"], "all"])
    ap.add_argument("--units", nargs="+", default=cfg["units"], choices=UNITS)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: only the first N utterances per dataset (writes to <encoder>__smokeN)")
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--device", default=cfg["extraction"]["device"])
    ap.add_argument("--overwrite", action="store_true", help="discard an existing cache for this encoder")
    ap.add_argument("--verify", action="store_true", help="check an existing cache instead of extracting")
    ap.add_argument("--dry-run", action="store_true", help="build and validate the unit tables, print sizes, exit (no torch)")
    a = ap.parse_args(argv)

    t_start = time.time()
    tables = build_tables(a.units, cfg["l2_window_half_s"])
    utts = utterance_list(tables, a.limit)
    print(f"[ssl] {len(utts)} utterances, units: {', '.join(tables)}")
    _describe(tables)
    if a.dry_run:
        print("[ssl] dry run OK: every Sprint 3 meta row resolved to exactly one pooling span")
        return
    if not a.encoder:
        ap.error("--encoder is required unless --dry-run")

    root = cache_root(cfg, a.cache_root)
    names = list(cfg["encoders"]) if a.encoder == "all" else [a.encoder]
    for key in names:
        spec = cfg["encoders"][key]
        out_dir = root / (f"{key}__smoke{a.limit}" if a.limit else key)
        if a.overwrite and out_dir.exists():
            import shutil
            shutil.rmtree(out_dir)
        if a.verify:
            m = json.loads((out_dir / "manifest.json").read_text())
            rep = verify_cache(out_dir, tables, m["dim"], m["n_layers"], utts if a.limit else None)
            print(f"[ssl] verify {out_dir} (complete={m['stats']['complete']})\n{json.dumps(rep, indent=2)}")
            if not m["stats"]["complete"] or any(not r["shape_ok"] or r["non_finite_rows"] or r["unfilled_rows"] for r in rep.values()):
                raise SystemExit(1)
            continue
        enc = FrozenEncoder(spec["hf_id"], spec.get("revision"), a.device)
        fp = {"code": CODE_VERSION, "encoder": key, "hf_id": spec["hf_id"], "revision": spec.get("revision"), "commit": enc.commit,
              "cache_dtype": cfg["extraction"]["cache_dtype"], "l2_half_s": cfg["l2_window_half_s"], "units": list(tables),
              "keys_sha256": {n: t.keys_sha256 for n, t in tables.items()}, "limit": a.limit}
        print(f"[ssl] {key}: {spec['hf_id']} @ {enc.commit} on {enc.device}; {enc.n_layers} layers x {enc.dim}; -> {out_dir}")
        stats = run_extraction(tables, utts, enc, load_wav, out_dir, n_layers=enc.n_layers, dim=enc.dim,
                               shard_size=cfg["extraction"]["shard_size"], cache_dtype=cfg["extraction"]["cache_dtype"],
                               fingerprint=fp, conv=(enc.kernel, enc.stride), progress=True)
        import torch, transformers
        manifest = {**fp, "n_layers": enc.n_layers, "dim": enc.dim, "normalises_input": enc.normalizes, "device": enc.device,
                    "rows": {n: len(t.df) for n, t in tables.items()}, "stats": stats, "git_sha": _git_sha(),
                    "torch": torch.__version__, "transformers": transformers.__version__, "seconds": round(time.time() - t_start, 1)}
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"[ssl] {key}: {'complete' if stats['complete'] else 'INCOMPLETE'} - {stats['utterances']} utterances this run, "
              f"{stats['shards_done']}/{stats['shards_total']} shards")


if __name__ == "__main__":
    main(sys.argv[1:])
