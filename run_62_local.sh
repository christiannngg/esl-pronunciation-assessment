#!/usr/bin/env bash
# US 6.2 (Sprint 6) on the lab machine. Resumable: re-running skips finished jobs.
#   bash run_62_local.sh [--parallel N] <command>
#
#   test                 unit tests of the diagnosis code (CPU, seconds)
#   analyse              the diagnosis from what already exists: transfer, gradient shares / conflict, stopping compromise, ablations that have been run
#                        -> results/tables/multitask/us62_report.md (+ us62_*.csv, results/figures/multitask/us62_gradients_<group>.png)   NO GPU, a few minutes
#   ablate <name>|all    SO762 ablation jobs (GPU), tag mt62_<name>, 3 seeds: phone_pair | word_phone | w_mispron3
#   status               which ablations are finished
#
# Order: run `analyse` first (needs the US 6.1 headline csvs: report_us61 --tag mt_base --groups so762 ; report_us61 --tag mt_lr --variant lr --groups l2),
# read it, then run the ablations, then `analyse` again (it picks up every finished ablation).
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
FAILED=logs/62_failed.txt; mkdir -p logs; : > "$FAILED"
export FAILED
ABLATIONS="phone_pair word_phone w_mispron3"

# one work unit: repeat the call until it reports nothing left to fit (--budget 3000 s stops a call early)
unit () {
  name="$1"; shift
  log="logs/62_${name}.log"
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
    echo "running $# job(s), up to $PAR at a time (logs/62_*.log; watch the GPU with: watch -n 2 nvidia-smi)"
    printf '%s\n' "$@" | xargs -P "$PAR" -I{} bash -c 'set -f; unit {}'
  else
    for u in "$@"; do bash -c 'set -f; unit '"$u"; done
  fi
}

abl_units () {                           # $1 = ablation name
  for seed in 0 1 2; do echo "mt62_${1}_s${seed} python -m src.models.multitask run --tag mt62_$1 --variant $1 --groups so762 --seeds $seed --budget 3000"; done
}

case "${1:-}" in
  test)
    python -m pytest tests/test_report_us62.py tests/test_multitask.py tests/test_report_us61.py -q ;;
  analyse|analyze)
    shift; python -m src.evaluation.report_us62 "$@" ;;
  ablate)
    names="${*:2}"
    [ "$names" = "all" ] && names="$ABLATIONS"
    [ -z "$names" ] && { echo "usage: bash run_62_local.sh [--parallel N] ablate phone_pair|word_phone|w_mispron3|all  (several names allowed)"; exit 1; }
    units=()
    for a in $names; do
      case " $ABLATIONS " in *" $a "*) ;; *) echo "unknown ablation $a (choose from: $ABLATIONS)"; exit 1 ;; esac
      mapfile -t u < <(abl_units "$a"); units+=("${u[@]}")
    done
    run_units "${units[@]}"
    for a in $names; do echo "== $a"; python -m src.models.multitask status --tag "mt62_$a" --variant "$a" --groups so762; done
    [ -s "$FAILED" ] && { echo "some jobs failed:"; cat "$FAILED"; } ;;
  status)
    for a in $ABLATIONS; do echo "== $a"; python -m src.models.multitask status --tag "mt62_$a" --variant "$a" --groups so762; done ;;
  *) echo "usage: bash run_62_local.sh [--parallel N] test|analyse|ablate <name>|all|status"; exit 1 ;;
esac
