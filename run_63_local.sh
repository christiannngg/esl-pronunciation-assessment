#!/usr/bin/env bash
# US 6.3 (Sprint 6) on the lab GPU: fixes for negative transfer / task imbalance in the SO762 multi-task model. Resumable: re-running skips finished jobs.
#   bash run_63_local.sh [--parallel N] <command>
#
#   test                    unit tests of the new code (loss weighters, report rules, config)   (CPU, seconds)
#   smoke                   2-epoch jobs of uncertainty and gradnorm_a05 (speed / memory / the weights move)   -> tag mt63_smoke_<name>, safe to delete
#   train <name>|all        SO762 jobs (GPU), tag mt63_<name>, 3 seeds: sent4 | uncertainty | gradnorm_a05 | gradnorm_a15
#   status                  which candidates are finished (also shows mt_base and mt62_word_phone, which US 6.3 reuses)
#   report                  -> results/tables/multitask/us63_report.md (+ us63_*.csv). NO GPU. Needs: mt_base (US 6.1), mt62_word_phone (US 6.2), base (US 5.1)
#
# Order: test -> smoke -> train all (or one name at a time) -> report.
# The regroup candidate = sent4 (new) + word_phone (already trained in US 6.2, tag mt62_word_phone; `bash run_62_local.sh ablate word_phone` if missing).
# The adoption rule (configs/multitask.yaml compare63) was fixed before any of these jobs ran; do not edit it after looking at results.
cd "$(dirname "$0")"
source .venv/bin/activate
export SSL_CACHE="${SSL_CACHE:-$HOME/chrisg/ssl_cache}"
PAR=1
if [ "${1:-}" = "--parallel" ]; then PAR="${2:-1}"; shift 2; fi
case "$PAR" in ''|*[!0-9]*|0) echo "--parallel needs a positive integer"; exit 1 ;; esac
NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( NCPU / PAR > 0 ? NCPU / PAR : 1 ))}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PAR
FAILED=logs/63_failed.txt; mkdir -p logs; : > "$FAILED"
export FAILED
NAMES="sent4 uncertainty gradnorm_a05 gradnorm_a15"

# one work unit: repeat the call until it reports nothing left to fit (--budget 3000 s stops a call early)
unit () {
  name="$1"; shift
  log="logs/63_${name}.log"
  [ "$PAR" -gt 1 ] && echo "[start] $name"
  for i in $(seq 1 40); do
    if [ "$PAR" -gt 1 ]; then out=$("$@" 2>&1); rc=$?; echo "$out" >> "$log"
    else out=$("$@" 2>&1); rc=$?; echo "$out" | tee -a "$log" | tail -n 12; fi
    if [ $rc -ne 0 ]; then echo "FAILED $name (rc=$rc), see $log" | tee -a "$FAILED"; echo "$out" | tail -n 25; return 1; fi
    echo "$out" | grep -q "finished: 0 fitted" && break
  done
  [ "$PAR" -gt 1 ] && echo "[done]  $name"
  return 0
}
export -f unit

run_units () {
  if [ "$PAR" -gt 1 ]; then
    echo "running $# job(s), up to $PAR at a time (logs/63_*.log; watch the GPU with: watch -n 2 nvidia-smi)"
    printf '%s\n' "$@" | xargs -P "$PAR" -I{} bash -c 'set -f; unit {}'
  else
    for u in "$@"; do bash -c 'set -f; unit '"$u"; done
  fi
}

train_units () {                         # $1 = name
  for seed in 0 1 2; do echo "mt63_${1}_s${seed} python -m src.models.multitask run --tag mt63_$1 --variant $1 --groups so762 --seeds $seed --budget 3000"; done
}

case "${1:-}" in
  test)
    python -m pytest tests/test_us63.py tests/test_multitask.py tests/test_report_us61.py tests/test_report_us62.py -q ;;
  smoke)
    for n in uncertainty gradnorm_a05; do
      python -m src.models.multitask run --tag "mt63_smoke_$n" --variant "$n" --groups so762 --folds 0 --seeds 0 --epochs 2 --force
    done ;;
  train)
    names="${*:2}"
    [ "$names" = "all" ] && names="$NAMES"
    [ -z "$names" ] && { echo "usage: bash run_63_local.sh [--parallel N] train sent4|uncertainty|gradnorm_a05|gradnorm_a15|all  (several names allowed)"; exit 1; }
    units=()
    for a in $names; do
      case " $NAMES " in *" $a "*) ;; *) echo "unknown candidate $a (choose from: $NAMES)"; exit 1 ;; esac
      mapfile -t u < <(train_units "$a"); units+=("${u[@]}")
    done
    run_units "${units[@]}"
    for a in $names; do echo "== $a"; python -m src.models.multitask status --tag "mt63_$a" --variant "$a" --groups so762; done
    [ -s "$FAILED" ] && { echo "some jobs failed:"; cat "$FAILED"; } ;;
  status)
    echo "== full (US 6.1)";                  python -m src.models.multitask status --tag mt_base --groups so762
    echo "== word_phone (US 6.2, regroup)";   python -m src.models.multitask status --tag mt62_word_phone --variant word_phone --groups so762
    for a in $NAMES; do echo "== $a"; python -m src.models.multitask status --tag "mt63_$a" --variant "$a" --groups so762; done ;;
  report)
    shift; python -m src.evaluation.report_us63 "$@" ;;
  *) echo "usage: bash run_63_local.sh [--parallel N] test|smoke|train <name>|all|status|report"; exit 1 ;;
esac
