"""
checkpoints.py — Manage the ECGFounder 1-lead checkpoint.

Single source of truth for:
  - the canonical checkpoint URL + path
  - downloading the checkpoint if it's missing
  - building and loading the model in one call

Public API:
    ensure_checkpoint(path=CHECKPOINT_PATH) -> str
    load_ecgfounder(device, ckpt_path=None)  -> Net1D
    download_checkpoint(url, out_file, num_chunks=8)  # low-level
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import requests
import torch

from net1d import Net1D


# ── Canonical configuration ─────────────────────────────────────────────────

CHECKPOINT_URL: str = (
    "https://huggingface.co/PKUDigitalHealth/ECGFounder/"
    "resolve/main/1_lead_ECGFounder.pth"
)
CHECKPOINT_PATH: str = "./checkpoint/1_lead_ECGFounder.pth"

# 1-lead checkpoint is ~118 MB. Treat any file under 100 MB as incomplete.
MIN_CHECKPOINT_BYTES: int = 100 * 1024 * 1024

# Net1D architecture that matches the published 1-lead ECGFounder weights.
ECGFOUNDER_NET1D_CONFIG = dict(
    in_channels=1,
    base_filters=64,
    ratio=1,
    filter_list=[64, 160, 160, 400, 400, 1024, 1024],
    m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
    kernel_size=16,
    stride=2,
    groups_width=16,
    verbose=False,
    use_bn=False,
    use_do=False,
    n_classes=150,
)


# ── Download ────────────────────────────────────────────────────────────────

def download_checkpoint(
    url: str,
    out_file: str,
    num_chunks: int = 8,
    quiet: bool = False,
) -> None:
    """Parallel-chunk download to `out_file`. Overwrites if it exists.

    Raises RuntimeError if any chunk fails or the resulting file is smaller
    than MIN_CHECKPOINT_BYTES.
    """
    def _say(msg: str) -> None:
        if not quiet:
            print(msg)

    _say(f"Resolving final URL for {url} …")
    try:
        r = requests.head(url, allow_redirects=True, timeout=15)
        final_url = r.url
        content_length = int(r.headers.get('content-length', 0))
    except Exception as e:
        _say(f"HEAD request failed ({e}); falling back to GET …")
        r = requests.get(url, stream=True, allow_redirects=True, timeout=15)
        final_url = r.url
        content_length = int(r.headers.get('content-length', 0))

    if content_length == 0:
        raise RuntimeError(
            f"Could not determine content length for {url}. "
            f"Refusing to download into an unsized file."
        )

    _say(f"Final URL:  {final_url}")
    _say(f"File size:  {content_length / (1024 * 1024):.2f} MB")

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    # Pre-allocate so chunks can write to disjoint offsets concurrently.
    with open(out_file, "wb") as f:
        f.truncate(content_length)

    chunk_size = content_length // num_chunks

    def _download_chunk(chunk_id: int) -> bool:
        try:
            start = chunk_id * chunk_size
            end = (
                start + chunk_size - 1
                if chunk_id < num_chunks - 1
                else content_length - 1
            )
            headers = {"Range": f"bytes={start}-{end}"}
            _say(f"  chunk {chunk_id + 1}/{num_chunks}  ({start}..{end})")

            res = requests.get(final_url, headers=headers, stream=True, timeout=60)
            if res.status_code not in (200, 206):
                _say(f"  chunk {chunk_id + 1} failed: HTTP {res.status_code}")
                return False

            with open(out_file, "r+b") as f:
                f.seek(start)
                for block in res.iter_content(chunk_size=1024 * 1024):
                    if block:
                        f.write(block)
            return True
        except Exception as e:
            _say(f"  chunk {chunk_id + 1} error: {e}")
            return False

    _say(f"Downloading with {num_chunks} parallel connections …")
    with ThreadPoolExecutor(max_workers=num_chunks) as executor:
        results = list(executor.map(_download_chunk, range(num_chunks)))

    if not all(results):
        raise RuntimeError(
            f"One or more chunks failed downloading {url}. "
            f"Partial file left at {out_file}."
        )

    actual_size = os.path.getsize(out_file)
    if actual_size < MIN_CHECKPOINT_BYTES:
        raise RuntimeError(
            f"Downloaded file is suspiciously small ({actual_size} bytes < "
            f"{MIN_CHECKPOINT_BYTES} byte minimum). "
            f"The download may have been truncated. Removing {out_file}."
        )

    _say(f"Saved checkpoint → {out_file}  ({actual_size / (1024 * 1024):.2f} MB)")


def ensure_checkpoint(
    path: str = CHECKPOINT_PATH,
    url: str = CHECKPOINT_URL,
    quiet: bool = False,
) -> str:
    """Return `path`, downloading the checkpoint if it doesn't already exist.

    Treats files smaller than MIN_CHECKPOINT_BYTES as incomplete and re-downloads.
    Returns the path on success; raises RuntimeError on download failure.
    """
    if os.path.exists(path) and os.path.getsize(path) >= MIN_CHECKPOINT_BYTES:
        if not quiet:
            print(f"[checkpoints] using existing {path} "
                  f"({os.path.getsize(path) / (1024*1024):.1f} MB)")
        return path

    if not quiet:
        if os.path.exists(path):
            print(f"[checkpoints] {path} is too small "
                  f"({os.path.getsize(path)} bytes); re-downloading …")
        else:
            print(f"[checkpoints] {path} not found; downloading from HuggingFace …")

    download_checkpoint(url, path, quiet=quiet)
    return path


# ── Model loader ────────────────────────────────────────────────────────────

def load_ecgfounder(
    device,
    ckpt_path: Optional[str] = None,
    *,
    download_if_missing: bool = True,
) -> Net1D:
    """Build the ECGFounder Net1D and load the 1-lead weights.

    Args:
        device: torch device (or string) to load the model onto.
        ckpt_path: Override path; defaults to CHECKPOINT_PATH.
        download_if_missing: If True (default), auto-download via
                             `ensure_checkpoint()` if the file is absent.
                             Set False to fail fast on missing weights.

    Returns:
        Net1D model in eval() mode, on `device`.
    """
    path = ckpt_path or CHECKPOINT_PATH
    if download_if_missing:
        ensure_checkpoint(path)
    elif not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found at {path}. "
            f"Run `python3 fast_download.py` or pass download_if_missing=True."
        )

    model = Net1D(**ECGFOUNDER_NET1D_CONFIG)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    model.load_state_dict(sd)
    return model.to(device).eval()
