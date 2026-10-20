#!/usr/bin/env bash
# US 5.1 (Sprint 5) on the lab GPU. Resumable: re-running skips finished jobs.
#   bash run_51_local.sh verify   # fine-tuning pooling == cached frozen embeddings (needs the Sprint 4 cache); run this FIRST
#   bash run_51_local.sh smoke    # 1 sentence + 1 phone job, 2 epochs, to check speed / memory before the real run
#   bash run_51_local.sh train    # all 9 tasks, all folds, all seeds (tag base)
#   bash run_51_local.sh report   # results/tables/finetune/us51_report.md
cd "$(dirname "$0")"
source .venv/bin/activate
export SSL_CACHE="${SSL_CACHE:-$HOME/chrisg/ssl_cache}"
TAG="${TAG:-base}"
TASKS="sent_accuracy sent_fluency sent_prosodic sent_total word_accuracy phone_score phone_mispron_t1.0 l2_phone_error l2_word_error"
FAILED=logs/51_failed.txt; mkdir -p logs; : > "$FAILED"

case "${1:-}" in
  verify)
    python -m src.models.finetune dry-run
    python -m src.models.finetune verify --n 20 --cache-root "$SSL_CACHE" ;;
  smoke)
    python -m src.models.finetune run --tag smoke --tasks sent_accuracy phone_mispron_t1.0 --folds 0 --seeds 0 --epochs 2 --force
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv ;;
  train)
    for t in $TASKS; do                  # one task at a time so a failure is attributable; --budget 3000 stops a call early, loop continues
      for i in $(seq 1 40); do
        out=$(python -m src.models.finetune run --tag "$TAG" --tasks "$t" --budget 3000 2>&1); rc=$?
        echo "$out" | tail -n 4
        if [ $rc -ne 0 ]; then echo "FAILED $t (rc=$rc)" | tee -a "$FAILED"; echo "$out" | tail -n 25; break; fi
        echo "$out" | grep -q "finished: 0 fitted" && break
      done
      python -m src.models.finetune status --tag "$TAG" | grep "^$t "
    done ;;
  report)
    python -m src.evaluation.report_us51 --tag "$TAG" ;;
  *) echo "usage: bash run_51_local.sh verify|smoke|train|report"; exit 1 ;;
esac
