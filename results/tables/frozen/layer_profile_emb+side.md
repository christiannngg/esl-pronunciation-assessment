# Per-layer linear probe (emb+side) - layer chosen on VALIDATION, test shown at that layer

Regression: RMSE (lower is better). Classification: macro-F1 (higher is better). One linear head, one seed per layer; layer 0 = conv front end.

## sent_accuracy

| encoder         | best layer (val) | val   | test  | layer 12 val | layer 12 test |
| --------------- | ---------------- | ----- | ----- | ------------ | ------------- |
| wavlm_base_plus | 12               | 1.128 | 1.295 | 1.128        | 1.295         |
| hubert_base     | 10               | 1.091 | 1.295 | 1.170        | 1.332         |
| wav2vec2_base   | 9                | 1.150 | 1.289 | 1.207        | 1.412         |

## word_accuracy

| encoder         | best layer (val) | val   | test  | layer 12 val | layer 12 test |
| --------------- | ---------------- | ----- | ----- | ------------ | ------------- |
| wavlm_base_plus | 12               | 1.715 | 1.687 | 1.715        | 1.687         |
| hubert_base     | 10               | 1.736 | 1.697 | 1.766        | 1.689         |
| wav2vec2_base   | 1                | 1.722 | 1.705 | 1.750        | 1.724         |

## phone_score

| encoder         | best layer (val) | val   | test  | layer 12 val | layer 12 test |
| --------------- | ---------------- | ----- | ----- | ------------ | ------------- |
| wavlm_base_plus | 12               | 0.355 | 0.349 | 0.355        | 0.349         |
| hubert_base     | 8                | 0.360 | 0.353 | 0.366        | 0.351         |
| wav2vec2_base   | 0                | 0.363 | 0.357 | 0.369        | 0.356         |

## phone_mispron_t1.0

| encoder         | best layer (val) | val   | test  | layer 12 val | layer 12 test |
| --------------- | ---------------- | ----- | ----- | ------------ | ------------- |
| wavlm_base_plus | 12               | 0.564 | 0.572 | 0.564        | 0.572         |
| hubert_base     | 9                | 0.556 | 0.558 | 0.548        | 0.571         |
| wav2vec2_base   | 12               | 0.564 | 0.576 | 0.564        | 0.576         |

## l2_phone_error

| encoder         | best layer (val) | val   | test  | layer 12 val | layer 12 test |
| --------------- | ---------------- | ----- | ----- | ------------ | ------------- |
| wavlm_base_plus | 12               | 0.358 | 0.358 | 0.358        | 0.358         |
| hubert_base     | 12               | 0.340 | 0.341 | 0.340        | 0.341         |
| wav2vec2_base   | 9                | 0.330 | 0.328 | 0.289        | 0.281         |

## l2_word_error

| encoder         | best layer (val) | val   | test  | layer 12 val | layer 12 test |
| --------------- | ---------------- | ----- | ----- | ------------ | ------------- |
| wavlm_base_plus | 11               | 0.711 | 0.707 | 0.699        | 0.698         |
| hubert_base     | 11               | 0.688 | 0.685 | 0.679        | 0.678         |
| wav2vec2_base   | 9                | 0.688 | 0.685 | 0.620        | 0.612         |
