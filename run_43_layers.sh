#!/usr/bin/env bash
cd "$(dirname "$0")"; source .venv/bin/activate
export SSL_CACHE="${SSL_CACHE:-$HOME/chrisg/ssl_cache}"
ENCODERS="wavlm_base_plus hubert_base wav2vec2_base"
FAILED=logs/43_layers_failed.txt; : > "$FAILED"
step () { echo "=== $(date +%T) $*"; "$@" || echo "FAILED: $*" | tee -a "$FAILED"; }

for enc in $ENCODERS; do   # 16b: probes for the three sentence tasks
  step python -m src.models.frozen_heads run --encoder $enc --probe --layers 6-12 \
       --tasks sent_fluency sent_prosodic sent_total --budget 3000 --cache-root "$SSL_CACHE"
done
for enc in $ENCODERS; do   # 16c: shortlist, then the full protocol (twice: the 2nd pass is the resume check)
  step python -m src.models.frozen_fix layers --encoder $enc --dry-run --cache-root "$SSL_CACHE"
  step python -m src.models.frozen_fix layers --encoder $enc --budget 3000 --cache-root "$SSL_CACHE"
  step python -m src.models.frozen_fix layers --encoder $enc --budget 3000 --cache-root "$SSL_CACHE"
done
for enc in $ENCODERS; do   # 16g again: the source head must see the layer systems now
  step python -m src.models.frozen_fix mispron --encoder $enc --force --cache-root "$SSL_CACHE"
done
# 16h (optional): train-score refill for wav2vec2
step python -m src.models.frozen_heads run --encoder wav2vec2_base --stage final --force \
     --tasks phone_mispron_t1.0 l2_phone_error l2_word_error --budget 3000 --cache-root "$SSL_CACHE"
echo "=== ALL DONE $(date +%T); failures:"; cat "$FAILED"
