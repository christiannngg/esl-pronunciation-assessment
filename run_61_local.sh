#!/usr/bin/env bash
# US 6.1 (Sprint 6) on the lab GPU. Resumable: re-running skips finished jobs.
#   bash run_61_local.sh test              # unit tests for the multi-task code (CPU, a few seconds)
#   bash run_61_local.sh dry               # build both groups on the REAL tables: heads must agree on utterances and splits; prints sizes (no GPU)
#   bash run_61_local.sh smoke             # 2-epoch job of each group (SO762 7 heads, L2-ARCTIC 2 heads) to check speed / memory before the real run
#   bash run_61_local.sh st-l2             # extra single-task L2-ARCTIC seeds 1, 2 (comparators; 2 tasks x 4 folds x 2 seeds, for `base` and for the US 5.3 selection)
#   bash run_61_local.sh train base        # multi-task recipe `base` (tag mt_base): both groups, all folds, all seeds
#   bash run_61_local.sh train lr          # multi-task recipe `lr`   (tag mt_lr; the US 5.3 `lr` recipe on the shared encoder)
#   bash run_61_local.sh screen            # which recipe goes forward (validation only) -> results/tables/multitask/us61_recipe_screen.md
#   bash run_61_local.sh report [tag]      # results/tables/multitask/us61_report_<tag>.md  (default tag: mt_base)
#   bash run_61_local.sh status [tag]
cd "$(dirname "$0")"
source .venv/bin/activate
export SSL_CACHE="${SSL_CACHE:-$HOME/chrisg/ssl_cache}"
FAILED=logs/61_failed.txt; mkdir -p logs; : > "$FAILED"
L2_TASKS="l2_phone_error l2_word_error"

train_group () {                         # $1 = tag, $2 = variant ("" = base recipe), $3 = group
  tag="$1"; variant="$2"; group="$3"
  vflag=(); [ -n "$variant" ] && vflag=(--variant "$variant")
  for i in $(seq 1 40); do               # --budget 3000 stops a call early, the loop continues until nothing is left
    out=$(python -m src.models.multitask run --tag "$tag" "${vflag[@]}" --groups "$group" --budget 3000 2>&1); rc=$?
    echo "$out" | tail -n 12
    if [ $rc -ne 0 ]; then echo "FAILED $tag $group (rc=$rc)" | tee -a "$FAILED"; echo "$out" | tail -n 25; break; fi
    echo "$out" | grep -q "finished: 0 fitted" && break
  done
  python -m src.models.multitask status --tag "$tag" "${vflag[@]}" --groups "$group"
}

st_l2 () {                               # single-task L2 comparators with seeds 1, 2: tag <variant>_l2x for `base` and for every variant US 5.3 selected for an L2 task
  variants=$(python - <<'EOF'
import pandas as pd
sel = pd.read_csv("results/tables/finetune/us53_selection.csv").set_index("task")["selected"]
print(" ".join(sorted({"base", *[sel[t] for t in ("l2_phone_error", "l2_word_error") if t in sel.index]})))
EOF
)
  for v in $variants; do
    vflag=(); [ "$v" != "base" ] && vflag=(--variant "$v")
    for t in $L2_TASKS; do
      if [ "$v" != "base" ] && ! python - "$t" "$v" <<'EOF'
import sys, pandas as pd
sel = pd.read_csv("results/tables/finetune/us53_selection.csv").set_index("task")["selected"]
sys.exit(0 if sel.get(sys.argv[1]) == sys.argv[2] else 1)
EOF
      then continue; fi                  # a variant's extra seeds are needed only for the tasks it was selected for
      for i in $(seq 1 40); do
        out=$(python -m src.models.finetune run --tag "${v}_l2x" "${vflag[@]}" --tasks "$t" --seeds 1 2 --budget 3000 2>&1); rc=$?
        echo "$out" | tail -n 4
        if [ $rc -ne 0 ]; then echo "FAILED ${v}_l2x $t (rc=$rc)" | tee -a "$FAILED"; echo "$out" | tail -n 25; break; fi
        echo "$out" | grep -q "finished: 0 fitted" && break
      done
    done
  done
}

case "${1:-}" in
  test)
    python -m pytest tests/test_multitask.py tests/test_report_us61.py tests/test_finetune.py -q ;;
  dry)
    python -m src.models.multitask dry-run ;;
  smoke)
    python -m src.models.multitask run --tag smoke_mt --groups so762 --folds 0 --seeds 0 --epochs 2 --force
    python -m src.models.multitask run --tag smoke_mt --groups l2 --folds 0 --seeds 0 --epochs 2 --force
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv ;;
  st-l2)
    st_l2 ;;
  train)
    case "${2:-}" in
      base) for g in so762 l2; do train_group mt_base "" "$g"; done ;;
      lr)   for g in so762 l2; do train_group mt_lr lr "$g"; done ;;
      *) echo "usage: bash run_61_local.sh train base|lr"; exit 1 ;;
    esac ;;
  screen)
    python -m src.evaluation.report_us61 --recipe-screen --tags mt_base mt_lr ;;
  report)
    python -m src.evaluation.report_us61 --tag "${2:-mt_base}" ;;
  status)
    python -m src.models.multitask status --tag "${2:-mt_base}" ;;
  *) echo "usage: bash run_61_local.sh test|dry|smoke|st-l2|train base|train lr|screen|report [tag]|status [tag]"; exit 1 ;;
esac
