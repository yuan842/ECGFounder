"""State-dict surgery: merge selected classifier rows from a fine-tuned
checkpoint into the base checkpoint.

Two variants supported:

  --variant B1 (default):  base backbone + base head, with rows in --active-idx
                           replaced by the corresponding rows from --fine-ckpt.
                           ⇒ fzark inactive heads are bit-identical to base.

  --variant B2:            fine-tuned backbone + fine-tuned head, with rows
                           NOT in --active-idx replaced by base.
                           ⇒ active heads + fuzzy backbone preserved.

Default --active-idx is the 10 PTB-XL classes with positives in the
fuzzylead2 training data: 2, 5, 6, 18, 26, 32, 36, 62, 70, 82.

The classifier layer is named ``dense`` in Net1D
(``dense.weight`` (150, 1024) and ``dense.bias`` (150,)).
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

DEFAULT_BASE   = "checkpoint/1_lead_ECGFounder.pth"
DEFAULT_FINE   = "checkpoint/1_lead_ECGFounder_fuzzy.pth"
DEFAULT_ACTIVE = [2, 5, 6, 18, 26, 32, 36, 62, 70, 82]


def _load_sd(path):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    return obj['state_dict'] if isinstance(obj, dict) and 'state_dict' in obj else obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-ckpt', default=DEFAULT_BASE)
    ap.add_argument('--fine-ckpt', default=DEFAULT_FINE)
    ap.add_argument('--out-ckpt',  required=True)
    ap.add_argument('--variant',   choices=['B1', 'B2'], default='B1')
    ap.add_argument('--active-idx', type=int, nargs='+', default=DEFAULT_ACTIVE)
    args = ap.parse_args()

    print(f"Loading base ckpt: {args.base_ckpt}")
    base = _load_sd(args.base_ckpt)
    print(f"Loading fine ckpt: {args.fine_ckpt}")
    fine = _load_sd(args.fine_ckpt)

    # Sanity: both checkpoints must have matching shapes for dense.{weight,bias}
    bw, bb = base['dense.weight'], base['dense.bias']
    fw, fb = fine['dense.weight'], fine['dense.bias']
    assert bw.shape == fw.shape == (150, 1024), (bw.shape, fw.shape)
    assert bb.shape == fb.shape == (150,)

    active = torch.tensor(args.active_idx, dtype=torch.long)
    print(f"Active head indices ({len(active)}): {args.active_idx}")

    if args.variant == 'B1':
        # Start from base, overwrite active rows with fine-tuned values
        merged = {k: v.clone() for k, v in base.items()}
        merged['dense.weight'][active] = fw[active]
        merged['dense.bias'][active]   = fb[active]
        # Sanity: backbone identical to base
        backbone_diff = sum(
            (merged[k] - base[k]).abs().sum().item()
            for k in merged if k not in ('dense.weight', 'dense.bias')
        )
        assert backbone_diff == 0, f"B1 backbone drift: {backbone_diff}"
    else:  # B2
        merged = {k: v.clone() for k, v in fine.items()}
        # Inactive rows = everything not in active
        all_idx = torch.arange(150)
        inactive = torch.tensor(
            [i for i in all_idx.tolist() if i not in args.active_idx],
            dtype=torch.long,
        )
        merged['dense.weight'][inactive] = bw[inactive]
        merged['dense.bias'][inactive]   = bb[inactive]

    # Confirm classifier-row diff against the two sources
    n_active_diff_w = (merged['dense.weight'] != base['dense.weight']).any(dim=1).sum().item()
    n_active_diff_b = (merged['dense.bias']  != base['dense.bias']).sum().item()
    print(f"Classifier rows differing from BASE: weight={n_active_diff_w}  bias={n_active_diff_b}")
    n_match_fine_w = (merged['dense.weight'] == fine['dense.weight']).all(dim=1).sum().item()
    n_match_fine_b = (merged['dense.bias']  == fine['dense.bias']).sum().item()
    print(f"Classifier rows matching FUZZY:    weight={n_match_fine_w}  bias={n_match_fine_b}")

    os.makedirs(os.path.dirname(args.out_ckpt) or '.', exist_ok=True)
    torch.save(merged, args.out_ckpt)
    print(f"Wrote {args.out_ckpt}  ({os.path.getsize(args.out_ckpt)/1e6:.1f} MB)")


if __name__ == '__main__':
    main()
