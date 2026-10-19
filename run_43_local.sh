#!/usr/bin/env bash
# US 4.3 steps 16e-16g, local GPU. Resumable: re-running skips finished jobs.
cd "$(dirname "$0")"
source .venv/bin/activate
export SSL_CACHE="${SSL_CACHE:-$HOME/chrisg/ssl_cache}"
ENCODERS="wavlm_base_plus hubert_base wav2vec2_base"
FAILED=logs/43_failed.txt; : > "$FAILED"

# run one stage until nothing is left to fit (--budget 3000 stops a call early; loop and continue)
run_stage () {   # enc recipe stage
  for i in 1 2 3 4 5 6; do
    out=$(python -m src.models.frozen_fix run --encoder "$1" --recipe "$2" --stage "$3" \
          --budget 3000 --cache-root "$SSL_CACHE" 2>&1); rc=$?
    echo "$out" | tail -n 3
    if [ $rc -ne 0 ]; then echo "FAILED $1 $2 $3 (rc=$rc)" | tee -a "$FAILED"; echo "$out" | tail -n 25; return 1; fi
    echo "$out" | grep -q "finished: 0 fitted" && return 0
  done
}

for enc in $ENCODERS; do               # 16e: sentence recipes
  for recipe in sent_ws sent_ws_pros sent_avgpca sent_avgpca_pros; do
    echo "=== $(date +%T) $enc $recipe"
    run_stage "$enc" "$recipe" search && run_stage "$enc" "$recipe" final
    python -m src.models.frozen_fix status --encoder "$enc" --recipe "$recipe" --cache-root "$SSL_CACHE" | grep final
  done
done

for enc in $ENCODERS; do               # 16f: word recipes
  for recipe in word_mean_pros word_max_pros word_meanmax_pros word_meanmaxstd_pros; do
    echo "=== $(date +%T) $enc $recipe"
    run_stage "$enc" "$recipe" search && run_stage "$enc" "$recipe" final
    python -m src.models.frozen_fix status --encoder "$enc" --recipe "$recipe" --cache-root "$SSL_CACHE" | grep final
  done
done

for enc in $ENCODERS; do               # 16g: mispron out-of-fold probabilities
  echo "=== $(date +%T) $enc mispron"
  python -m src.models.frozen_fix mispron --encoder "$enc" --cache-root "$SSL_CACHE" \
    || echo "FAILED $enc mispron" | tee -a "$FAILED"
done
echo "=== ALL DONE $(date +%T); failures:"; cat "$FAILED"
