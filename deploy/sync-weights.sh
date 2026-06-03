#!/usr/bin/env bash
# Copy the gitignored runtime model weights to a deployment host.
#
# These artifacts are NOT in git (the no-weights rule) and are NOT all
# auto-downloaded from Hugging Face, so a fresh `git clone` on the server is
# missing them and the AFib evaluation (ScopedDetector) fails with e.g.
#   FileNotFoundError: 'res/scope_overlay/fuzzySL.pth'
#
# Run this from a LOCAL clone that already has the weights:
#   deploy/sync-weights.sh ubuntu@<ec2-host> [ssh-key] [remote-dir]
#
# Defaults: ssh-key=~/.ssh/ecg-demo, remote-dir=~/ECGFounder
#
# What it copies:
#   checkpoint/1_lead_ECGFounder_fuzzy.pth   (~353 MB, locally fine-tuned — NOT on HF)
#   res/scope_overlay/*.pth                  (~6 KB each, L1 projection heads)
# The base checkpoint/1_lead_ECGFounder.pth is auto-downloaded on first run
# (see SETUP.md C.4), so it is intentionally NOT synced here.
set -euo pipefail

HOST="${1:?usage: sync-weights.sh user@host [ssh-key] [remote-dir]}"
KEY="${2:-$HOME/.ssh/ecg-demo}"
REMOTE_DIR="${3:-~/ECGFounder}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SSH="ssh -i $KEY -o BatchMode=yes"

FUZZY="checkpoint/1_lead_ECGFounder_fuzzy.pth"
OVERLAY_GLOB="res/scope_overlay/*.pth"

# Preflight: confirm the local artifacts exist before touching the host.
[ -f "$FUZZY" ] || { echo "ERROR: missing local $FUZZY — train/obtain it first." >&2; exit 1; }
shopt -s nullglob
OVERLAY_FILES=( $OVERLAY_GLOB )
shopt -u nullglob
[ ${#OVERLAY_FILES[@]} -gt 0 ] || { echo "ERROR: no local $OVERLAY_GLOB found." >&2; exit 1; }

echo "→ ensuring remote dirs exist on $HOST"
$SSH "$HOST" "mkdir -p $REMOTE_DIR/checkpoint $REMOTE_DIR/res/scope_overlay"

echo "→ copying fuzzy backbone ($(du -h "$FUZZY" | cut -f1))"
scp -i "$KEY" -o BatchMode=yes "$FUZZY" "$HOST:$REMOTE_DIR/checkpoint/"

echo "→ copying ${#OVERLAY_FILES[@]} L1 overlay weight(s)"
scp -i "$KEY" -o BatchMode=yes "${OVERLAY_FILES[@]}" "$HOST:$REMOTE_DIR/res/scope_overlay/"

echo "→ verifying on host"
$SSH "$HOST" "ls -lh $REMOTE_DIR/checkpoint/1_lead_ECGFounder_fuzzy.pth $REMOTE_DIR/res/scope_overlay/*.pth"

echo "✓ weights synced. Restart the service:  sudo systemctl restart ecg-demo"
