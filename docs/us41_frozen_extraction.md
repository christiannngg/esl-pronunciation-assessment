# US 4.1: frozen-encoder embedding extraction (Sprint 4)

Code: `src/features/ssl_embeddings.py` · config: `configs/frozen.yaml` · tests: `tests/test_ssl_embeddings.py` ·
Colab runner: `notebooks/03_frozen_extraction_colab.ipynb` · data pack: `experiments/scripts/pack_colab_data.py`

## What it produces

For each encoder (`wav2vec2_base` = facebook/wav2vec2-base, `hubert_base` = facebook/hubert-base-ls960,
`wavlm_base_plus` = microsoft/wavlm-base-plus; all pretrain-only, 12 layers, 768-d) one directory
`<cache_root>/<encoder>/` with a float16 array per Sprint 3 unit and **all 13 hidden states** (index 0 = encoder input after
the conv front end, 1..12 = transformer layers):

| file | shape | pooling |
|---|---|---|
| `so762_sentence_X.npy` | (5000, 13, 1536) | mean ‖ std over every frame of the file |
| `so762_word_X.npy` | (31816, 13, 768) | mean over the MFA word interval |
| `so762_phone_X.npy` | (94445, 13, 768) | mean over the MFA phone interval |
| `l2_word_X.npy` | (33985, 13, 768) | mean over the span of the word's non-addition units |
| `l2_phone_X.npy` | (120243, 13, 768) | mean over the fixed ±0.075 s window around the annotated unit midpoint |

Roughly 6 GB per encoder. `manifest.json` records the model id and resolved hub commit, torch/transformers versions, row counts,
a SHA-256 of each unit's key columns and the git commit; `progress.json` lets an interrupted run resume.

## Row alignment with Sprint 3 (the property the comparison depends on)

Row *k* of `<unit>_X.npy` is row *k* of `data/processed/features/tables/<unit>_meta.csv.gz`, the table the MFCC baselines used. The
meta tables hold keys and labels but no timestamps, so times are joined from the alignment tables by key (`utt_id`, `word_idx`,
`phone_idx`) and the build fails if any row has no interval, an interval is duplicated, or a span is non-positive. Heads must load
labels, speakers and folds from the same meta file and may assert `keys_sha256` against the manifest.

## Design decisions (and why)

* **Frame → unit rule.** Frames are 20 ms apart with a 25 ms window, so frame *i* is centred at 12.5 ms + 20 ms·*i*. A unit pools the
  frames whose centre lies inside it; a unit with none (phones under ~20 ms, 3.2% of SO762 phones are ≤ 30 ms) takes the nearest frame.
* **L2-ARCTIC leakage control kept.** Phones use the same fixed window as the baseline, so the annotated duration, which gives away
  deletions and additions, is not visible to the heads. The window half-width is read from `configs/baseline.yaml`, not duplicated.
* **Batch size 1.** wav2vec 2.0 base was trained without an attention mask, so padded batches change its outputs. One utterance per
  forward pass avoids the issue; the cost is minutes per encoder on a T4.
* **Per-checkpoint input normalisation** comes from each checkpoint's own feature extractor.
* **Pooling in float64**, cumulative-sum based, then cast to float16 (finite check on every write; max |value| is in the manifest).
* **Silence is never trimmed**, same as Sprint 2/3.

## Running it

```bash
python -m src.features.ssl_embeddings --dry-run                     # no GPU: validates every unit table against the Sprint 3 meta
python -m src.features.ssl_embeddings --encoder wav2vec2_base --limit 20    # smoke test
python -m src.features.ssl_embeddings --encoder all                 # full run, resumable
python -m src.features.ssl_embeddings --encoder hubert_base --verify
python -m pytest tests/test_ssl_embeddings.py
```

On Colab use the notebook; set `DATA_ROOT` (unpacked pack) and `SSL_CACHE_ROOT` (local disk, not Drive).

## Known limits

* Hidden states depend on the checkpoint's receptive field (25 ms), so very short phones inherit context from neighbours; this is
  inherent to frozen frame features and is one of the things the US 4.2 diagnosis looks at.
* SO762 boundaries come from forced alignment of the canonical pronunciation (Sprint 2 notes the long-phone caveat).
* GPU kernels are not bit-reproducible across hardware; expect differences at the fp16 rounding level between runs on different GPUs.
