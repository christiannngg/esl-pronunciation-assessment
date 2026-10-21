#!/usr/bin/env bash
# US 5.3 (Sprint 5) on the lab GPU. Resumable: re-running skips finished jobs. Each variant is a tag of the same name (configs/finetune.yaml `variants`).
#   bash run_53_local.sh smoke            # 2-epoch check that the masking / noise / freezing variants run (a few minutes)
#   bash run_53_local.sh train lr         # one variant, all 9 tasks (about as long as US 5.1: ~1 h; freeze variants are faster)
#   bash run_53_local.sh train all        # lr reg mask noise freeze6, then freeze9 on the W6 tasks only (several hours, resumable)
#   bash run_53_local.sh report           # results/tables/finetune/us53_report.md (variants that have finished jobs)
cd "$(dirname "$0")"
source .venv/bin/activate
export SSL_CACHE="${SSL_CACHE:-$HOME/chrisg/ssl_cache}"
ALL_TASKS="sent_accuracy sent_fluency sent_prosodic sent_total word_accuracy phone_score phone_mispron_t1.0 l2_phone_error l2_word_error"
W6_TASKS="sent_prosodic phone_score l2_word_error word_accuracy"
FAILED=logs/53_failed.txt; mkdir -p logs; : > "$FAILED"

train_variant () {                       # $1 = variant, $2... = tasks
  v="$1"; shift
  for t in "$@"; do
    for i in $(seq 1 40); do
      out=$(python -m src.models.finetune run --tag "$v" --variant "$v" --tasks "$t" --budget 3000 2>&1); rc=$?
      echo "$out" | tail -n 4
      if [ $rc -ne 0 ]; then echo "FAILED $v $t (rc=$rc)" | tee -a "$FAILED"; echo "$out" | tail -n 25; break; fi
      echo "$out" | grep -q "finished: 0 fitted" && break
    done
    python -m src.models.finetune status --tag "$v" --variant "$v" | grep "^$t "
  done
}

case "${1:-}" in
  smoke)
    for v in mask noise freeze6 lr reg; do
      python -m src.models.finetune run --tag "smoke_$v" --variant "$v" --tasks sent_accuracy phone_mispron_t1.0 --folds 0 --seeds 0 --epochs 2 --force || echo "FAILED smoke $v" | tee -a "$FAILED"
    done
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv ;;
  train)
    case "${2:-}" in
      all) for v in lr reg mask noise freeze6; do train_variant "$v" $ALL_TASKS; done
           train_variant freeze9 $W6_TASKS ;;
      freeze9) train_variant freeze9 $W6_TASKS ;;
      lr|reg|mask|noise|freeze6) train_variant "$2" $ALL_TASKS ;;
      *) echo "usage: bash run_53_local.sh train lr|reg|mask|noise|freeze6|freeze9|all"; exit 1 ;;
    esac ;;
  report)
    python -m src.evaluation.report_us53 ;;
  *) echo "usage: bash run_53_local.sh smoke|train <variant|all>|report"; exit 1 ;;
esac
