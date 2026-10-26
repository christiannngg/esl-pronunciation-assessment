#!/usr/bin/env bash
# US 6.1 (Sprint 6) on the lab GPU. Resumable: re-running skips finished jobs.
#   bash run_61_local.sh [--parallel N] <command>
#
#   test                     unit tests for the multi-task code (CPU, a few seconds)
#   dry                      build both groups on the REAL tables: heads must agree on utterances and splits; prints sizes (no GPU)
#   smoke                    2-epoch job of each group (SO762 7 heads, L2-ARCTIC 2 heads): speed and memory
#   smoke --parallel 3       the same SO762 smoke job with seeds 0,1,2 AT THE SAME TIME: compare s/epoch and peak GPU GB with the single run to pick N
#   st-l2                    extra single-task L2-ARCTIC seeds 1, 2 (comparators for `base` and for the US 5.3 selection)
#   train base | train lr    multi-task recipes (tags mt_base, mt_lr): both groups, all folds, all seeds
#   screen                   which recipe goes forward (validation only) -> results/tables/multitask/us61_recipe_screen.md
#   report [tag]             results/tables/multitask/us61_report_<tag>.md  (default tag: mt_base)
#   status [tag]
#
# --parallel N : run up to N jobs at once on the same GPU, each as its own process. A job is one (group, seed) for `train`, one (variant, task, seed) for
# `st-l2`; different jobs write different files, so results are identical to a sequential run (same seeds, same recipe, same batches). Pick N from the
# `smoke --parallel` output: peak GPU GB x N must fit in the card's memory, and total time must drop (the GPU may already be near full use with one job).
# Each job's output goes to logs/61_<...>.log when N > 1; progress lines are printed here.
cd "$(dirname "$0")"
source .venv/bin/activate
export SSL_CACHE="${SSL_CACHE:-$HOME/chrisg/ssl_cache}"
PAR=1
if [ "${1:-}" = "--parallel" ]; then PAR="${2:-1}"; shift 2; fi
case "$PAR" in ''|*[!0-9]*|0) echo "--parallel needs a positive integer"; exit 1 ;; esac
# N processes each running numpy / torch CPU threads would fight for the cores: split them
NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( NCPU / PAR > 0 ? NCPU / PAR : 1 ))}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PAR
FAILED=logs/61_failed.txt; mkdir -p logs; : > "$FAILED"
export FAILED
L2_TASKS="l2_phone_error l2_word_error"
SEL=results/tables/finetune/us53_selection.csv

# ---- one work unit: repeat the call until it reports nothing left to fit (--budget 3000 s stops a call early) -------------------------------------------
# usage: unit <log-name> <command words...>     (stdout/err -> terminal when PAR=1, -> logs/61_<log-name>.log otherwise)
unit () {
  name="$1"; shift
  log="logs/61_${name}.log"
  [ "$PAR" -gt 1 ] && echo "[start] $name"
  once=0; case "$name" in smoke_*) once=1 ;; esac      # smoke jobs use --force: a single call, never repeated
  for i in $(seq 1 40); do
    if [ "$PAR" -gt 1 ]; then out=$("$@" 2>&1); rc=$?; echo "$out" >> "$log"
    else out=$("$@" 2>&1); rc=$?; echo "$out" | tee -a "$log" | tail -n 12; fi
    if [ $rc -ne 0 ]; then echo "FAILED $name (rc=$rc), see $log" | tee -a "$FAILED"; echo "$out" | tail -n 25; return 1; fi
    [ "$once" = 1 ] && break
    echo "$out" | grep -q "finished: 0 fitted" && break
  done
  [ "$PAR" -gt 1 ] && echo "[done]  $name"
  return 0
}
export -f unit

# ---- run a list of units, at most $PAR at a time. Each unit is one line: "<log-name> <command words...>" ---------------------------------------------
run_units () {
  if [ "$PAR" -gt 1 ]; then
    echo "running $# job(s), up to $PAR at a time (logs/61_*.log; watch the GPU with: watch -n 2 nvidia-smi)"
    printf '%s\n' "$@" | xargs -P "$PAR" -I{} bash -c 'set -f; unit {}'
  else
    for u in "$@"; do bash -c 'set -f; unit '"$u"; done
  fi
}

# the multi-task jobs of one recipe, SO762 first (the heavier group) so the long jobs start early
mt_units () {                            # $1 = tag, $2 = variant or ""
  tag="$1"; variant="$2"; vflag=""; [ -n "$variant" ] && vflag="--variant $variant"
  for seed in 0 1 2; do echo "${tag}_so762_s${seed} python -m src.models.multitask run --tag $tag $vflag --groups so762 --seeds $seed --budget 3000"; done
  for seed in 0 1 2; do echo "${tag}_l2_s${seed} python -m src.models.multitask run --tag $tag $vflag --groups l2 --seeds $seed --budget 3000"; done
}

# single-task L2 comparators, seeds 1 and 2: tag <variant>_l2x for `base` and for every variant US 5.3 selected for an L2 task
l2x_units () {
  for t in $L2_TASKS; do
    v=$(python - "$t" <<'EOF'
import sys, pandas as pd
print(pd.read_csv("results/tables/finetune/us53_selection.csv").set_index("task")["selected"].get(sys.argv[1], "base"))
EOF
)
    for tag_variant in base "$v"; do
      vflag=""; [ "$tag_variant" != "base" ] && vflag="--variant $tag_variant"
      for seed in 1 2; do echo "${tag_variant}_l2x_${t}_s${seed} python -m src.models.finetune run --tag ${tag_variant}_l2x $vflag --tasks $t --seeds $seed --budget 3000"; done
    done
  done | sort -u
}

case "${1:-}" in
  test)
    python -m pytest tests/test_multitask.py tests/test_report_us61.py tests/test_finetune.py -q ;;
  dry)
    python -m src.models.multitask dry-run ;;
  smoke)
    if [ "$PAR" -gt 1 ]; then                  # N copies of the same 2-epoch SO762 job, different seeds and tags -> compare s/epoch and peak GPU with N=1
      units=(); for s in $(seq 0 $((PAR - 1))); do units+=("smoke_par_s${s} python -m src.models.multitask run --tag smoke_mt_par${s} --groups so762 --folds 0 --seeds $s --epochs 2 --force"); done
      ( while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >> logs/61_smoke_gpu.csv 2>/dev/null; sleep 5; done ) & sampler=$!
      start=$(date +%s); run_units "${units[@]}"; echo "wall time for $PAR concurrent jobs: $(( $(date +%s) - start )) s"
      kill $sampler 2>/dev/null
      echo "peak GPU memory / utilisation samples (memory.used MiB, utilisation %): logs/61_smoke_gpu.csv"; sort -t, -k1 -n logs/61_smoke_gpu.csv 2>/dev/null | tail -n 1
      grep -h "peak GPU\|best_epoch" logs/61_smoke_par_s*.log
    else
      python -m src.models.multitask run --tag smoke_mt --groups so762 --folds 0 --seeds 0 --epochs 2 --force
      python -m src.models.multitask run --tag smoke_mt --groups l2 --folds 0 --seeds 0 --epochs 2 --force
      nvidia-smi --query-gpu=memory.used,memory.total --format=csv
    fi ;;
  st-l2)
    mapfile -t units < <(l2x_units); run_units "${units[@]}"; python -m src.models.finetune status --tag base_l2x ;;
  train)
    case "${2:-}" in
      base) mapfile -t units < <(mt_units mt_base ""); run_units "${units[@]}"; python -m src.models.multitask status --tag mt_base ;;
      lr)   mapfile -t units < <(mt_units mt_lr lr);   run_units "${units[@]}"; python -m src.models.multitask status --tag mt_lr --variant lr ;;
      *) echo "usage: bash run_61_local.sh [--parallel N] train base|lr"; exit 1 ;;
    esac
    [ -s "$FAILED" ] && { echo "some jobs failed:"; cat "$FAILED"; } ;;
  screen)
    python -m src.evaluation.report_us61 --recipe-screen --tags mt_base mt_lr ;;
  report)
    python -m src.evaluation.report_us61 --tag "${2:-mt_base}" ;;
  status)
    python -m src.models.multitask status --tag "${2:-mt_base}" ;;
  *) echo "usage: bash run_61_local.sh [--parallel N] test|dry|smoke|st-l2|train base|train lr|screen|report [tag]|status [tag]"; exit 1 ;;
esac
