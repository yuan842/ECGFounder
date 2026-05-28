"""
fast_download.py — CLI wrapper to download the ECGFounder 1-lead checkpoint.

Underlying logic lives in `checkpoints.py`. This script exists for the historical
entry point referenced in setup docs:

    python3 fast_download.py

Eval scripts call `checkpoints.ensure_checkpoint()` automatically via
`checkpoints.load_ecgfounder()`, so explicit invocation is rarely needed.
"""
from checkpoints import ensure_checkpoint


if __name__ == "__main__":
    ensure_checkpoint()
    print("Model checkpoint ready.")
