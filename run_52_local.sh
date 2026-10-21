#!/usr/bin/env bash
# US 5.2 (Sprint 5) on the lab GPU. Resumable.
#   bash run_52_local.sh seen        # diagnostic runs: 20 % of every training speaker's utterances held back (tag base__seen), all tasks / folds / seeds
#   bash run_52_local.sh diagnose    # needs tag base (US 5.1) and base__seen; writes results/tables/finetune/us52_report.md + figures
cd "$(dirname "$0")"
source .venv/bin/activate
SEEN_TAG="${SEEN_TAG:-base__seen}"
TASKS="sent_accuracy sent_fluency sent_prosodic sent_total word_accuracy phone_score phone_mispron_t1.0 l2_phone_error l2_word_error"
FAILED=logs/52_failed.txt; mkdir -p logs; : > "$FAILED"

case "${1:-}" in
  seen)
    for t in $TASKS; do
      for i in $(seq 1 40); do
        out=$(python -m src.models.finetune run --tag "$SEEN_TAG" --holdout-seen 0.2 --tasks "$t" --budget 3000 2>&1); rc=$?
        echo "$out" | tail -n 3
        if [ $rc -ne 0 ]; then echo "FAILED $t (rc=$rc)" | tee -a "$FAILED"; echo "$out" | tail -n 25; break; fi
        echo "$out" | grep -q "finished: 0 fitted" && break
      done
      python -m src.models.finetune status --tag "$SEEN_TAG" | grep "^$t "
    done ;;
  diagnose)
    python -m src.evaluation.report_us51 --tag base > /dev/null     # refreshes us51_headline.csv / us51_jobs.csv that the diagnosis reads
    python -m src.evaluation.diagnose_finetune --tag base --seen-tag "$SEEN_TAG" ;;
  *) echo "usage: bash run_52_local.sh seen|diagnose"; exit 1 ;;
esac
