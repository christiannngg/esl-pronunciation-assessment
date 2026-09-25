"""
Environment smoke test — run this on EACH machine (Colab, school GPU
cluster, local Mac) after setting up the environment, to confirm:
  1. Core packages import correctly
  2. GPU/CUDA is visible to PyTorch (where applicable)
  3. W&B is authenticated and can log a run

Usage:
    python experiments/scripts/smoke_test.py
"""

import sys


def check_imports():
    print("Checking core imports...")
    import torch  # noqa: F401
    import torchaudio  # noqa: F401
    import transformers  # noqa: F401
    import librosa  # noqa: F401
    import sklearn  # noqa: F401
    import wandb  # noqa: F401
    print("  OK — torch, torchaudio, transformers, librosa, sklearn, wandb all import.")


def check_gpu():
    import torch
    available = torch.cuda.is_available()
    print(f"Checking GPU access... torch.cuda.is_available() = {available}")
    if available:
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    else:
        print("  WARNING: No GPU detected. Expected on a CPU-only local machine; "
              "should be True on Colab (GPU runtime) and the school cluster.")
    return available


def check_wandb():
    from src.utils.tracking import init_run
    print("Checking W&B logging...")
    run = init_run(job_type="smoke-test", tags=["smoke-test"])
    run.log({"smoke_test_ok": 1})
    run.finish()
    print(f"  OK — logged a test run to project '{run.project}'. Check your W&B dashboard.")


if __name__ == "__main__":
    check_imports()
    check_gpu()
    try:
        check_wandb()
    except Exception as e:
        print(f"  W&B check failed: {e}\n  Did you run `wandb login` on this machine?")
        sys.exit(1)
    print("\nSmoke test complete.")
