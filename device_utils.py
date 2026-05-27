"""
device_utils.py
===============

Project-wide PyTorch device-selection policy.

Default order: **MPS → CUDA → CPU** (Apple M-series prioritized).

Public API
----------
    default_device()           -> torch.device
    resolve_device(cli_value)  -> torch.device
    DEFAULT_DEVICE_STR         -> resolved string (set at import time)

Override
--------
Set environment variable `ECGFOUNDER_DEVICE` to one of {'mps', 'cuda',
'cuda:0', 'cpu', 'auto'} to force a non-default device project-wide.
'auto' applies the MPS→CUDA→CPU fallback chain.

Usage in scripts
----------------
    from device_utils import resolve_device
    device = resolve_device(args.device)   # respects CLI, env var, then default

This file is the SINGLE SOURCE OF TRUTH for device choice across the project.
"""
import os
import warnings
from typing import Optional

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# ─── Policy ────────────────────────────────────────────────────────────────
#   1. Honor an explicit caller request ('cpu', 'cuda', 'mps', 'cuda:0', ...).
#   2. Otherwise honor the ECGFOUNDER_DEVICE env var (same set + 'auto').
#   3. Otherwise apply the MPS → CUDA → CPU fallback chain.
DEFAULT_POLICY = "auto"   # current project default; can be changed here only

ENV_VAR = "ECGFOUNDER_DEVICE"


def _auto_select() -> str:
    """Return the first available device in the project's preferred order."""
    if not _HAS_TORCH:
        return "cpu"
    # 1. MPS — Apple Silicon (M-series) GPU acceleration
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except Exception:
        pass
    # 2. CUDA — NVIDIA GPU
    try:
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    # 3. CPU fallback
    return "cpu"


def resolve_device(cli_value: Optional[str] = None,
                   policy: Optional[str] = None) -> "torch.device":
    """
    Resolve the PyTorch device using the project policy.

    Parameters
    ----------
    cli_value : explicit device string from a CLI flag (e.g. argparse --device).
                Takes precedence over everything else when not None / empty.
    policy    : override the default policy ('auto', 'mps', 'cuda', 'cpu').
                Defaults to env var, then DEFAULT_POLICY.

    Returns
    -------
    torch.device
    """
    if not _HAS_TORCH:
        raise RuntimeError("torch is not installed")

    requested = (cli_value or os.environ.get(ENV_VAR) or policy
                 or DEFAULT_POLICY).strip().lower()

    if requested in ("", "auto"):
        return torch.device(_auto_select())

    # Explicit request — try, fall back loudly if unavailable
    if requested.startswith("mps"):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        warnings.warn("MPS requested but not available; falling back to "
                      "CUDA → CPU.")
        return torch.device(_auto_select())

    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested if ":" in requested else "cuda:0")
        warnings.warn("CUDA requested but not available; falling back to "
                      "MPS → CPU.")
        return torch.device(_auto_select())

    if requested == "cpu":
        return torch.device("cpu")

    # Unknown string — try torch's own parser; warn if it fails.
    try:
        return torch.device(requested)
    except Exception:
        warnings.warn(f"Unknown device '{requested}'; auto-selecting.")
        return torch.device(_auto_select())


def default_device() -> "torch.device":
    """The project's default device (MPS → CUDA → CPU)."""
    return resolve_device(None)


# Resolved at import time for quick reference / logging
DEFAULT_DEVICE_STR = _auto_select() if _HAS_TORCH else "cpu"


# ─── Helpful banner ────────────────────────────────────────────────────────
def banner(device=None) -> str:
    """Return a one-line summary suitable for prefixing CLI output."""
    if device is None:
        device = default_device()
    return f"[ECGFounder] device = {device}   (policy: MPS → CUDA → CPU)"


if __name__ == "__main__":
    print(banner())
