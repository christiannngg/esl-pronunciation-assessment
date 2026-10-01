"""
Shared data-path resolution across environments (local Mac, Colab,
school GPU cluster). Both datasets live in one Google Drive folder:

    MyDrive/esl-pronunciation-data/raw/speechocean762/
    MyDrive/esl-pronunciation-data/raw/l2arctic_release_v5.0/

On Colab, this mounts Drive automatically. On your local Mac (or the
school cluster, if it can reach Drive), set the DATA_ROOT environment
variable to wherever that Drive folder is synced locally, e.g. in your
shell profile:

    export DATA_ROOT="/Users/chirstiangonzalez/Google Drive/My Drive/esl-pronunciation-data"

If DATA_ROOT isn't set and we're not on Colab, this falls back to the
repo-local ./data directory (useful for quick local-only testing with
a small sample, but NOT where the full datasets should live long-term).
"""

import os
from pathlib import Path


def _is_colab() -> bool:
    return "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ


def get_data_root() -> Path:
    """
    Returns the root Path containing raw/ (and later processed/, external/)
    for both datasets, resolved appropriately for the current environment.
    """
    if _is_colab():
        drive_mount = Path("/content/drive")
        if not (drive_mount / "MyDrive").exists():
            from google.colab import drive  # type: ignore

            drive.mount(str(drive_mount))
        return drive_mount / "MyDrive" / "esl-pronunciation-data"

    env_root = os.environ.get("DATA_ROOT")
    if env_root:
        return Path(env_root)

    # Fallback: repo-local data/ (gitignored). Fine for quick local checks,
    # but set DATA_ROOT to your synced Drive folder for the real datasets.
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data"


def speechocean762_root() -> Path:
    return get_data_root() / "raw" / "speechocean762"


def l2arctic_root() -> Path:
    return get_data_root() / "raw" / "l2arctic_release_v5.0"