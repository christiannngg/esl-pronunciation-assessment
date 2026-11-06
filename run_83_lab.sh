#!/bin/bash
# US 8.3 (Sprint 8) on the lab GPU: the CV grid that selects the generalization fix. Resumable: finished jobs are skipped.
# Submit (one GPU job; replace the stage):
#   sbatch --gres=gpu:1 -c 8 --mem=32G -t 04:00:00 --output=s8_%x-%j.out --job-name=cv run_83_lab.sh cv all
# Stages:
#   test                  unit tests of the US 8.3 code (CPU, seconds)
#   aug [--limit N]       child-like copies of the adult training audio (CPU; 2 banks) -> $S8_AUG_ROOT
#   smoke                 2-epoch jobs on CV fold 0: f0, f1_s10 (and f2_a if its bank exists)  -> tags s8smoke_*, safe to delete
#   cv <cand>...|all      5 folds x each candidate (f0 f1_s05 f1_s10 f2_a f2_b); `all` also builds the F2 banks in the background while F0/F1 train
#   status                which CV jobs are finished
#   select                -> results/tables/fix/us83_cv_selection.{csv,json}   (NO GPU; needs all CV jobs)
# The selection rule (docs/sprint8_plan.md section 5 + the two precisions logged in its section 9) was fixed before any of these jobs ran.
cd "$(dirname "$0")" || exit 1
source .venv/bin/activate
export SSL_CACHE="${SSL_CACHE:-$HOME/chrisg/ssl_cache}"
export S8_AUG_ROOT="${S8_AUG_ROOT:-/scratch/$USER/aug_s8}"
export PYTHONPATH=.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p logs results/splits_s8
FAILED=logs/83_failed.txt; touch "$FAILED"
ALL="f0 f1_s05 f1_s10 f2_a f2_b"

split_files () {      # fold files -> results/splits_s8/so762_cv_s8_fold<k>.csv (the 125 official test speakers are `excluded`)
  python - <<'PY'
from pathlib import Path
from src.models import fix_s8 as F
df = F.make_folds()
import pandas as pd
saved = pd.read_csv(F.CV_CSV, dtype={"speaker": str})
assert saved.equals(df), "data/splits/so762_cv_s8.csv differs from fix_s8.make_folds(): the fold file must be the committed one"
F.check_folds(saved)
for k in range(F.N_FOLDS):
    print(F.fold_split_csv(k, Path("results/splits_s8"), folds=saved))
PY
}

fit () {              # fit <tag> <variant> <fold> [extra args]
  tag="$1"; var="$2"; k="$3"; shift 3
  python -m src.models.multitask run --tag "$tag" --variant "$var" --groups so762 --folds 0 --seeds 0 \
         --split-csv "results/splits_s8/so762_cv_s8_fold${k}.csv" "$@" 2>&1 | tee -a "logs/83_${tag}.log" | tail -n 6
  rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && echo "FAILED $tag (rc=$rc), see logs/83_${tag}.log" | tee -a "$FAILED"
  return 0
}

train_cands () {
  for c in "$@"; do
    case " $ALL " in *" $c "*) ;; *) echo "unknown candidate $c (choose from: $ALL)"; exit 1 ;; esac
    for k in 0 1 2 3 4; do echo "=== $c fold $k"; fit "s8cv${k}_${c}" "s8_${c}" "$k"; done
  done
}

case "${1:-}" in
  test)
    python -m pytest tests/test_fix_s8.py tests/test_report_us83.py tests/test_multitask.py tests/test_us63.py -q ;;
  aug)
    shift; python experiments/scripts/precompute_aug_s8.py "$@" ;;
  smoke)
    split_files
    fit s8smoke_f0 s8_f0 0 --epochs 2 --force
    fit s8smoke_f1_s10 s8_f1_s10 0 --epochs 2 --force
    if [ -d "$S8_AUG_ROOT/shift_2_4" ]; then fit s8smoke_f2_a s8_f2_a 0 --epochs 2 --force; else echo "no F2 bank under $S8_AUG_ROOT: run the aug stage first"; fi
    cat "$FAILED" ;;
  cv)
    shift
    split_files
    if [ "${1:-}" = "all" ]; then
      python experiments/scripts/precompute_aug_s8.py --cores 6 > logs/83_aug.log 2>&1 &
      AUG=$!
      train_cands f0 f1_s05 f1_s10
      wait "$AUG" || echo "FAILED aug (see logs/83_aug.log)" | tee -a "$FAILED"
      train_cands f2_a f2_b
    else
      train_cands "$@"
    fi
    echo "== failures:"; cat "$FAILED" ;;
  status)
    for c in $ALL; do for k in 0 1 2 3 4; do
      f="experiments/results/multitask/s8cv${k}_${c}/phone_mispron_t1.0/fold0/seed0.npz"; [ -f "$f" ] && echo "done  s8cv${k}_${c}" || echo "MISSING s8cv${k}_${c}"
    done; done ;;
  select)
    shift; python -m src.evaluation.report_us83 "$@" ;;
  *) echo "usage: bash run_83_lab.sh test|aug|smoke|cv <cand>...|all|status|select"; exit 1 ;;
esac
