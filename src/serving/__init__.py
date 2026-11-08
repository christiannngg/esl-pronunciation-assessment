"""Serving code for the Sprint 9 demo app.

PyTorch must be imported BEFORE scikit-learn / pandas / librosa on macOS: otherwise two OpenMP runtimes are loaded and the encoder load segfaults
(observed 2026-11-07, `python -m src.serving.parity`; `diag_load`, which imports torch first, works). Importing any ``src.serving`` module runs this first.
"""
try:
    import torch  # noqa: F401
except ImportError:                     # side_features / aligner do not need torch
    pass
