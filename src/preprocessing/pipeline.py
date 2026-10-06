"""
US 2.2 preprocessing entry point.

    python -m src.preprocessing.pipeline audio        # resample / normalise / QC
    python -m src.preprocessing.pipeline splits       # speaker lists -> data/splits/
    python -m src.preprocessing.pipeline mfa-prep     # corpora + SO762 dictionary + phone-set check
    python -m src.preprocessing.pipeline mfa-run      # run `mfa align` (needs MFA, i.e. your Mac env)
    python -m src.preprocessing.pipeline ingest       # MFA TextGrids -> tidy tables + validation report
    python -m src.preprocessing.pipeline all          # audio, splits, mfa-prep (stops before MFA)

Add ``--limit N`` to any stage for a small smoke test (first N utterances per dataset).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from src.data.loaders import load_l2arctic, load_so762
from src.preprocessing import alignment as al
from src.preprocessing import splits as sp
from src.preprocessing.audio import process_file
from src.preprocessing.labels import clean_l2arctic_phones
from src.utils.paths import get_data_root, speechocean762_root

REPO = Path(__file__).resolve().parents[2]
DEFAULTS = {"l2arctic_annotated_only": True, "mfa_beam": 100, "mfa_retry_beam": 400,
            "mfa_jobs": 4, "seed": sp.SEED, "so762_val_frac": 0.15}


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    path = REPO / "configs" / "preprocess.yaml"
    if path.exists():
        import yaml
        cfg.update(yaml.safe_load(path.read_text()) or {})
    return cfg


def proc_dir() -> Path:
    return get_data_root() / "processed"


# --------------------------------------------------------------------------- #
def _one(args):
    src, dst = args
    try:
        return process_file(Path(src), Path(dst)).as_dict()
    except Exception as e:  # reported, never silently dropped
        return {"error": f"{type(e).__name__}: {e}"}


def stage_audio(cfg, limit=None):
    so = load_so762(with_audio_info=False).utts
    l2 = load_l2arctic(with_audio_info=False).utts
    l2 = l2[(l2["corpus"] == "arctic") & (l2["has_annotation"] if cfg["l2arctic_annotated_only"] else True)]
    if limit:
        so, l2 = so.head(limit), l2.head(limit)
    out = proc_dir() / "audio"
    jobs = {"so762": [(r.wav_path, out / "so762" / f"{r.utt_id}.wav", r.utt_id, r.speaker) for r in so.itertuples()],
            "l2arctic": [(r.wav_path, out / "l2arctic" / r.speaker / f"{r.utt_id}.wav", r.utt_id, r.speaker)
                         for r in l2.itertuples()]}
    for ds, items in jobs.items():
        with ProcessPoolExecutor(max_workers=cfg["mfa_jobs"]) as ex:
            res = list(ex.map(_one, [(a, b) for a, b, _, _ in items], chunksize=32))
        qc = pd.DataFrame(res)
        qc.insert(0, "utt_id", [i[2] for i in items])
        qc.insert(1, "speaker", [i[3] for i in items])
        qc.to_csv(proc_dir() / f"audio_qc_{ds}.csv", index=False)
        errs = qc["error"].notna().sum() if "error" in qc else 0
        flags = {c: int(qc[c].sum()) for c in qc.columns if c.startswith("flag_")}
        print(f"[audio] {ds}: {len(qc)} files, {errs} errors, flags={flags}")


def stage_splits(cfg, limit=None):
    r = sp.build_all(seed=cfg["seed"])
    print(json.dumps({"so762_ok": r["so762"]["disjoint"] and r["so762"]["test_equals_official_test"],
                      "l2arctic_ok": r["l2arctic"]["all_checks_pass"]}))


# --------------------------------------------------------------------------- #
def _so762_items(limit=None):
    so = load_so762(with_audio_info=False).utts
    if limit:
        so = so.head(limit)
    return so


def stage_mfa_prep(cfg, limit=None):
    base = proc_dir() / "mfa"
    base.mkdir(parents=True, exist_ok=True)
    audio = proc_dir() / "audio"
    # SO762
    canon = al.so762_canonical(speechocean762_root() / "resource" / "scores.json")
    dic = al.build_so762_dictionary(canon)
    dict_path = base / "so762_lexicon.dict"
    al.write_mfa_dictionary(dic, dict_path)
    so = _so762_items(limit)
    n1 = al.prepare_corpus([(f"SPEAKER{r.speaker}", r.utt_id, audio / "so762" / f"{r.utt_id}.wav") for r in so.itertuples()],
                           base / "so762_corpus",
                           {u: al.utt_transcript(canon[u], dic) for u in so["utt_id"]}, raw=True)
    # L2-ARCTIC (annotated subset, for the consistency check)
    l2 = load_l2arctic(with_audio_info=False).utts
    l2 = l2[(l2["corpus"] == "arctic") & l2["has_annotation"]]
    if limit:
        l2 = l2.head(limit)
    n2 = al.prepare_corpus([(r.speaker, r.utt_id, audio / "l2arctic" / r.speaker / f"{r.utt_id}.wav") for r in l2.itertuples()],
                           base / "l2arctic_corpus", dict(zip(l2["utt_id"], l2["text"])))
    # phone-set check against the pretrained MFA dictionary (if downloaded)
    mfa_dict = Path.home() / "Documents" / "MFA" / "pretrained_models" / "dictionary" / "english_us_arpa.dict"
    info = {"so762_utts": n1, "l2arctic_utts": n2, "so762_dictionary_words": len(dic)}
    if mfa_dict.exists():
        inv = al.dictionary_phone_inventory(mfa_dict)
        ours = al.dictionary_phone_inventory(dict_path)
        info["phones_not_in_english_us_arpa"] = sorted(ours - inv)
        print(f"[mfa-prep] SO762 dictionary phones missing from english_us_arpa: {info['phones_not_in_english_us_arpa'] or 'none'}")
    else:
        print("[mfa-prep] english_us_arpa dictionary not found; run `mfa model download dictionary english_us_arpa`")
    (base / "mfa_prep.json").write_text(json.dumps(info, indent=2))
    print(f"[mfa-prep] corpora ready under {base}: {info}")


def stage_mfa_run(cfg, limit=None):
    if shutil.which("mfa") is None:
        sys.exit("`mfa` not found: activate the conda env that has montreal-forced-aligner (esl-pronunciation).")
    base = proc_dir() / "mfa"
    common = ["--beam", str(cfg["mfa_beam"]), "--retry_beam", str(cfg["mfa_retry_beam"]), "--clean",
              "--use_mp", "--num_jobs", str(cfg["mfa_jobs"]), "--output_format", "long_textgrid"]
    runs = [("so762_corpus", str(base / "so762_lexicon.dict"), "so762_aligned"),
            ("l2arctic_corpus", "english_us_arpa", "l2arctic_aligned")]
    for corpus, dictionary, out in runs:
        cmd = ["mfa", "align", str(base / corpus), dictionary, "english_us_arpa", str(base / out)] + common
        print("[mfa-run]", " ".join(cmd))
        subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------- #
def stage_ingest(cfg, limit=None):
    base = proc_dir() / "mfa"
    out = proc_dir() / "alignments"
    out.mkdir(parents=True, exist_ok=True)
    rep_dir = REPO / "results" / "tables" / "preprocessing"
    rep_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    # --- SO762: MFA -> word / phone tables ---
    canon = al.so762_canonical(speechocean762_root() / "resource" / "scores.json")
    so = _so762_items(limit)
    W, P, R = al.ingest_so762(base / "so762_aligned", canon, list(so["utt_id"]))
    W.to_csv(out / "so762_words.csv.gz", index=False)
    P.to_csv(out / "so762_phones.csv.gz", index=False)
    R.to_csv(out / "so762_alignment_status.csv", index=False)
    counts = R["status"].value_counts().to_dict()
    report["so762"] = {"status_counts": counts, "usable_fraction": float(counts.get("ok", 0) / max(len(R), 1)),
                       "rejection_reasons": R.get("reason", pd.Series(dtype=str)).dropna()
                       .str.replace(r"\d+", "N", regex=True).str.replace(r"'.*?'", "'w'", regex=True)
                       .value_counts().head(10).to_dict()}
    # who is rejected? (bias check: children / low scorers more often?)
    spk = pd.concat([so[["utt_id", "speaker", "age_group", "total"]]]).merge(R[["utt_id", "status"]], on="utt_id")
    report["so762"]["ok_rate_by_age_group"] = spk.groupby("age_group")["status"].apply(lambda s: float((s == "ok").mean())).round(4).to_dict()
    spk["total_band"] = pd.cut(spk["total"], [-1, 4, 6, 8, 10], labels=["0-4", "5-6", "7-8", "9-10"])
    report["so762"]["ok_rate_by_total_band"] = spk.groupby("total_band", observed=True)["status"].apply(lambda s: float((s == "ok").mean())).round(4).to_dict()

    # SO762 alignment sanity: phone duration distribution (frame-floor / degenerate alignments)
    dur = (P["end"] - P["start"]) * 1000
    report["so762"]["phone_duration_ms"] = {"median": float(dur.median()), "share_le_30ms": float((dur <= 30).mean()),
                                            "share_gt_500ms": float((dur > 500).mean()), "n_phones": int(len(P))}
    # --- L2-ARCTIC: cleaned manual-annotation labels + MFA consistency ---
    l2 = load_l2arctic(with_audio_info=False)
    ph = clean_l2arctic_phones(l2.phones)
    ph.to_csv(out / "l2arctic_phones_clean.csv.gz", index=False)
    report["l2arctic_labels"] = {"n_phone_rows": int(len(ph)), "n_label_repaired": int(ph["label_repaired"].sum()),
                                 "n_unmapped": int(ph["unmapped"].sum()),
                                 "error_class_counts": ph["error_class"].value_counts().to_dict()}
    utts = l2.utts[(l2.utts["corpus"] == "arctic") & l2.utts["has_annotation"]]
    provided = {r.utt_id: Path(r.wav_path).parent.parent / "textgrid" / f"{r.utt_id.split('_', 1)[1]}.TextGrid"
                for r in utts.itertuples()}
    report["l2arctic_annotation_vs_provided_fa"] = al.annotation_shift(ph[ph["utt_id"].isin(provided)], provided)
    if (base / "l2arctic_aligned").exists():
        report["l2arctic_mfa_vs_provided_fa"] = al.l2arctic_consistency(base / "l2arctic_aligned", provided)
        report["l2arctic_mfa_vs_provided_fa_words"] = al.word_boundary_consistency(base / "l2arctic_aligned", provided)
    else:
        report["l2arctic_mfa_vs_provided_fa"] = "MFA output not found; run mfa-run first"
    (rep_dir / "alignment_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str)[:3000])


STAGES = {"audio": stage_audio, "splits": stage_splits, "mfa-prep": stage_mfa_prep,
          "mfa-run": stage_mfa_run, "ingest": stage_ingest}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=list(STAGES) + ["all"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = load_config()
    for s in (["audio", "splits", "mfa-prep"] if a.stage == "all" else [a.stage]):
        STAGES[s](cfg, a.limit)


if __name__ == "__main__":
    main()
