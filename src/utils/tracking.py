"""
Shared Weights & Biases initialization helper.

Import this from any notebook or script (Colab, school GPU cluster, or local
Mac) so every run — regardless of where it's launched — lands in the same
W&B project with consistent naming and config logging.

Usage:
    from src.utils.tracking import init_run

    run = init_run(
        job_type="baseline-train",       # e.g. "eda", "baseline-train", "frozen-encoder", "finetune", "multitask"
        config={"model": "svm", "features": "mfcc"},
        tags=["sprint3", "traditional-ml"],
    )
    ...
    run.log({"val_mae": 0.42})
    run.finish()
"""

import os
import getpass
import socket

import wandb

PROJECT_NAME = "esl-pronunciation-assessment"


def _detect_environment() -> str:
    """Best-effort guess at which of the three machines this run is on."""
    if "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ:
        return "colab"
    hostname = socket.gethostname().lower()
    if "school" in hostname or "cluster" in hostname or "hpc" in hostname:
        return "school-cluster"
    return "local"


def init_run(job_type: str, config: dict | None = None, tags: list[str] | None = None):
    """
    Initialize a W&B run with consistent project naming and an
    auto-tagged environment (colab / school-cluster / local) so runs
    from different machines are easy to filter and compare.

    Requires `wandb login` to have been run once per machine
    (see README.md "Configure Weights & Biases").
    """
    env = _detect_environment()
    full_config = dict(config or {})
    full_config.setdefault("environment", env)
    full_config.setdefault("user", getpass.getuser())

    run = wandb.init(
        project=PROJECT_NAME,
        job_type=job_type,
        config=full_config,
        tags=(tags or []) + [env],
    )
    return run
