"""Tests for `checkpoints.ensure_checkpoint` and `checkpoints.load_ecgfounder`.

Network calls are mocked — no actual HuggingFace download happens.
"""
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from checkpoints import (
    CHECKPOINT_PATH,
    CHECKPOINT_URL,
    ECGFOUNDER_NET1D_CONFIG,
    MIN_CHECKPOINT_BYTES,
    download_checkpoint,
    ensure_checkpoint,
    load_ecgfounder,
)


# ─── ensure_checkpoint ──────────────────────────────────────────────────────

class TestEnsureCheckpoint:
    def test_existing_file_is_not_redownloaded(self, tmp_path):
        path = tmp_path / "dummy.pth"
        # Write a "large enough" placeholder
        path.write_bytes(b"\x00" * (MIN_CHECKPOINT_BYTES + 1024))

        with patch("checkpoints.download_checkpoint") as mock_dl:
            result = ensure_checkpoint(str(path), quiet=True)
            mock_dl.assert_not_called()
        assert result == str(path)

    def test_too_small_file_is_redownloaded(self, tmp_path):
        path = tmp_path / "dummy.pth"
        path.write_bytes(b"\x00" * 100)  # way under MIN_CHECKPOINT_BYTES

        with patch("checkpoints.download_checkpoint") as mock_dl:
            ensure_checkpoint(str(path), quiet=True)
            mock_dl.assert_called_once()

    def test_missing_file_triggers_download(self, tmp_path):
        path = tmp_path / "missing.pth"
        assert not path.exists()

        with patch("checkpoints.download_checkpoint") as mock_dl:
            ensure_checkpoint(str(path), quiet=True)
            mock_dl.assert_called_once()
            # Ensure URL + path are forwarded
            args, kwargs = mock_dl.call_args
            assert args[0] == CHECKPOINT_URL
            assert args[1] == str(path)


# ─── download_checkpoint (network mocked) ───────────────────────────────────

class TestDownloadCheckpoint:
    def _mock_response(self, status=200, body_size=MIN_CHECKPOINT_BYTES + 100):
        """Build a mock requests.Response that yields a body of size body_size."""
        resp = MagicMock()
        resp.status_code = status
        resp.url = "https://example.com/final"
        resp.headers = {"content-length": str(body_size)}
        # iter_content yields 1 MB blocks of zeros
        block = b"\x00" * (1024 * 1024)
        full_blocks = body_size // (1024 * 1024)
        rem = body_size - full_blocks * (1024 * 1024)
        chunks = [block] * full_blocks
        if rem:
            chunks.append(b"\x00" * rem)
        resp.iter_content = MagicMock(return_value=iter(chunks))
        return resp

    def test_successful_download_writes_full_file(self, tmp_path):
        out = tmp_path / "out.pth"
        body_size = MIN_CHECKPOINT_BYTES + 2048

        with patch("checkpoints.requests") as mock_requests:
            head_resp = self._mock_response(status=200, body_size=body_size)
            # Each chunked GET also returns a 206/200 response
            mock_requests.head.return_value = head_resp
            mock_requests.get.side_effect = lambda *a, **kw: self._mock_response(
                status=206, body_size=body_size
            )

            download_checkpoint(
                url="https://example.com/file.pth",
                out_file=str(out),
                num_chunks=2,
                quiet=True,
            )

        assert out.exists()
        # Pre-allocation pads to exactly content_length; cumulative writes may
        # extend slightly because each chunk overwrites its own range.
        assert os.path.getsize(str(out)) >= MIN_CHECKPOINT_BYTES

    def test_truncated_download_raises(self, tmp_path):
        out = tmp_path / "out.pth"
        # body_size below the sanity check
        body_size = 1024 * 1024  # 1 MB << MIN

        with patch("checkpoints.requests") as mock_requests:
            head_resp = self._mock_response(status=200, body_size=body_size)
            mock_requests.head.return_value = head_resp
            mock_requests.get.side_effect = lambda *a, **kw: self._mock_response(
                status=206, body_size=body_size
            )

            with pytest.raises(RuntimeError, match="suspiciously small"):
                download_checkpoint(
                    url="https://example.com/file.pth",
                    out_file=str(out),
                    num_chunks=2,
                    quiet=True,
                )

    def test_zero_content_length_raises(self, tmp_path):
        out = tmp_path / "out.pth"
        with patch("checkpoints.requests") as mock_requests:
            head_resp = MagicMock()
            head_resp.url = "https://example.com/final"
            head_resp.headers = {"content-length": "0"}
            mock_requests.head.return_value = head_resp
            mock_requests.get.return_value = head_resp  # fallback path also returns 0

            with pytest.raises(RuntimeError, match="content length"):
                download_checkpoint(
                    url="https://example.com/file.pth",
                    out_file=str(out),
                    quiet=True,
                )


# ─── load_ecgfounder ────────────────────────────────────────────────────────

class TestLoadEcgfounder:
    def test_fail_fast_when_missing_and_download_disabled(self, tmp_path):
        path = tmp_path / "missing.pth"
        with pytest.raises(FileNotFoundError, match="not found"):
            load_ecgfounder(
                device=torch.device("cpu"),
                ckpt_path=str(path),
                download_if_missing=False,
            )

    def test_loads_real_checkpoint_if_present(self):
        """Skip unless the canonical checkpoint is on disk locally."""
        if not (os.path.exists(CHECKPOINT_PATH)
                and os.path.getsize(CHECKPOINT_PATH) >= MIN_CHECKPOINT_BYTES):
            pytest.skip("Canonical checkpoint not available locally")

        model = load_ecgfounder(device=torch.device("cpu"))
        # Sanity: matches the config
        assert hasattr(model, "forward")
        # Smoke-test a forward pass shape
        x = torch.randn(1, 1, 5000)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 150)

    def test_config_contract_unchanged(self):
        """Guard against accidental changes to the ECGFounder config dict."""
        cfg = ECGFOUNDER_NET1D_CONFIG
        assert cfg["in_channels"] == 1
        assert cfg["n_classes"] == 150
        assert cfg["filter_list"] == [64, 160, 160, 400, 400, 1024, 1024]
        assert cfg["m_blocks_list"] == [2, 2, 2, 3, 3, 4, 4]
        assert cfg["kernel_size"] == 16
        assert cfg["stride"] == 2
