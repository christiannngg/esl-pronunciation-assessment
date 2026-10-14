"""
US 4.2 - failure-mode diagnosis of the frozen-encoder heads vs the Sprint 3 baseline (VALIDATION only).

    python experiments/scripts/diagnose_us42.py --encoders wavlm_base_plus hubert_base wav2vec2_base
    python experiments/scripts/diagnose_us42.py --encoders wavlm_base_plus --results-root experiments/results/frozen_outputs/frozen_results

Thin wrapper; the code is src/evaluation/diagnose_frozen.py (see its docstring for the tables). Outputs: results/analysis/us42/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.diagnose_frozen import main  # noqa: E402

if __name__ == "__main__":
    main()
