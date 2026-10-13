# US 4.2 support: layer probe, input ablation, train-score fix

Code: `src/models/frozen_heads.py` (`--probe`, `--layers`, `--force`) · `src/evaluation/report_frozen.py` (`--layer-profile`, `--ablation`) · config: `configs/frozen.yaml` → `heads.probe`.

## Per-layer linear probe (representation quality by depth)

One linear head, one seed per (encoder, layer, task), on six tasks (sent_accuracy, word_accuracy, phone_score, phone_mispron, l2_phone_error, l2_word_error).
The probe config is one of the main-grid configs (`wd0.01` for regression, `cwbalanced_wd0.001` for classification), so the layer-12 probe is the file the main search already wrote.
Layers are compared on **validation**; test is printed next to it and never used to pick a layer. Classification probes use balanced class weights in the loss and plain argmax (no decision weights), so curves are comparable across layers.

```bash
python -m src.models.frozen_heads run --encoder wavlm_base_plus --probe --layers 0-12        # Colab
python -m src.evaluation.report_frozen --encoders wavlm_base_plus hubert_base wav2vec2_base --layer-profile 0-12
```

Outputs: `results/tables/frozen/layer_profile_emb+side.{csv,md,png}`.

## Input ablation (is the gain from the embedding or from the side columns?)

```bash
python -m src.models.frozen_heads run --encoder wavlm_base_plus --mode emb  --stage search ; ... --stage final
python -m src.models.frozen_heads run --encoder wavlm_base_plus --mode side --stage search ; ... --stage final
python -m src.evaluation.report_frozen --encoders wavlm_base_plus --ablation
```

Each setting uses its own validation-selected head and variant. The table gives the paired speaker-bootstrap difference (full minus ablated) on identical test rows. Output: `results/tables/frozen/ablation_<encoder>_L12.{csv,md}`.

## Train score of the `cw+thr` rows

Earlier result files scored the train subsample with plain argmax, which is not the decision rule used on val / test, so the train column of `cw+thr` rows was not comparable. New `final` files also store `train_proba` / `train_y`; the report applies the validation-tuned decision weights to them. Rows from older files show NaN until the final stage is re-run with `--force` on the classification tasks.
