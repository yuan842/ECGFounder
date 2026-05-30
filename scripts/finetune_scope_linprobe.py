"""Scope-aligned linear probe — trains 6 of the 7 detection-scope heads
(idx 2, 4, 5, 6, 93, 98; Pause-142 excluded). Formerly finetune_6head_linprobe.py.

Target = the 7-label detection scope (label_config.scope_indices()) MINUS Pause
(142), which has 0 PTB-XL positives and cannot be trained here. So the 6 trainable
heads are: 2=NormalECG, 4=SinusBradycardia, 5=AFib, 6=SinusTachy, 93=SVT, 98=VT.
(Pause needs a non-PTB-XL source — MIMIC/fzark.) The "6" is the scope minus the one
untrainable head — not a fixed/independent target list.

Recipe:
  - Frozen Net1D backbone (all conv stages, requires_grad=False)
  - Frozen 144 non-target classifier rows (gradient blocked via register_hook)
  - Trainable: only the 6 target rows of dense.weight + dense.bias (6,150 params)
  - Loss: BCEWithLogitsLoss on the 6 target logits only
  - Data: PTB-XL at 4 derived-lead angles {45°, 60°, 75°, 90°}, split folds 1-8/9/10
  - Labels: full 150-vector PTB-XL labels (joined via ecg_id → filename)

Output checkpoint: checkpoint/1_lead_ECGFounder_6head_v2.pth
Regression gate: tested separately via eval_6head_regression.py.
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

from checkpoints import load_ecgfounder
from device_utils import resolve_device
import label_config as _L

# Configuration
# Probe target = the 7-label detection scope (label_config.scope_indices()),
# minus heads with no PTB-XL positives to train on. Pause (142) has 0 PTB-XL
# labels → untrainable here (needs MIMIC/fzark); everything else in scope is
# kept. Deriving from the scope keeps this aligned automatically with the rule.
PTBXL_UNTRAINABLE = {142}                      # Pause: 0 PTB-XL positives
TARGET_IDX   = sorted(_L.scope_indices() - PTBXL_UNTRAINABLE)   # [2, 4, 5, 6, 93, 98]
ALL_IDX      = list(range(150))
NON_TARGET   = [i for i in ALL_IDX if i not in TARGET_IDX]
ANGLES_TRAIN = (45, 60, 75, 90)
ANGLES_VAL   = (45, 60, 75, 90)
FUZZY_DIR    = Path("data/fuzzylead2/ptbxl")
PTBXL_CSV    = "csv/ptbxl_label.csv"
DEFAULT_OUT  = "checkpoint/1_lead_ECGFounder_6head_{suffix}.pth"
LOG_DIR      = Path("res/finetune_6head_v2")

# Map ecg_id -> filename helper (matches build_60deg_data.py)
def ecg_id_to_filename(eid: int) -> str:
    folder = (eid // 1000) * 1000
    return f"records500/{folder:05d}/{eid:05d}_hr"


def load_full_ptbxl_labels(ecg_ids: np.ndarray) -> np.ndarray:
    """For an array of ecg_ids, return the full (n, 150) PTB-XL label matrix
    by joining via filename_hr → label JSON."""
    print(f"  joining {len(ecg_ids)} ecg_ids to ptbxl_label.csv ...")
    csv = pd.read_csv(PTBXL_CSV).set_index('filename_hr')
    label_cache = {}
    out = np.zeros((len(ecg_ids), 150), dtype=np.float32)
    for i, eid in enumerate(ecg_ids):
        fn = ecg_id_to_filename(int(eid))
        if fn not in label_cache:
            row = csv.loc[fn]
            label_cache[fn] = np.array(json.loads(row['label']), dtype=np.float32)
        out[i] = label_cache[fn]
    return out


class MultiAngleDataset(Dataset):
    """Concatenates per-angle npz files; labels come from full PTB-XL CSV.

    Honors the recommended PTB-XL fold convention (ptbxl_splits):
      train → folds 1-8, val → fold 9  (both sourced from train_*deg.npz, which
                                         holds folds 1-9, then masked by ecg_id)
      test  → fold 10                   (the val_*deg.npz / val_60deg_split.npz)
    No patient crosses the train/val/test boundary (folds are patient-stratified).
    """
    def __init__(self, split: str, angles=ANGLES_TRAIN):
        import ptbxl_splits
        if split not in ("train", "val", "test"):
            raise ValueError(split)
        use_train_npz = split in ("train", "val")     # folds 1-9 live in train npz
        ecgs, ids, src_angles = [], [], []
        for a in angles:
            if use_train_npz:
                fn = FUZZY_DIR / f"train_{a}deg.npz"
            else:
                fn = FUZZY_DIR / (f"val_{a}deg.npz" if a != 60 else "val_60deg_split.npz")
            if not fn.exists():
                raise FileNotFoundError(f"{fn} not found — run scripts/build_60deg_data.py first")
            z = np.load(fn)
            eid = z['ecg_ids']
            keep = ptbxl_splits.mask_for(eid, split)   # fold-based partition (1-8 / 9 / 10)
            ecgs.append(z['ecg'][keep].astype(np.float32))
            ids.append(eid[keep])
            src_angles.extend([a] * int(keep.sum()))
        self.ecg = np.concatenate(ecgs, axis=0)
        self.ecg_ids = np.concatenate(ids, axis=0)
        self.angles  = np.array(src_angles, dtype=np.int32)

        # Resolve full PTB-XL labels (one cache lookup per unique ecg_id)
        self.labels = load_full_ptbxl_labels(self.ecg_ids)
        print(f"  {split}: ecg={self.ecg.shape}, labels={self.labels.shape}, angles={set(angles)}")
        # Per-target-head positive counts
        for i in TARGET_IDX:
            print(f"    idx {i:>3}: {int(self.labels[:, i].sum()):>5} positives")

    def __len__(self): return self.ecg.shape[0]
    def __getitem__(self, idx):
        return (torch.from_numpy(self.ecg[idx]),
                torch.from_numpy(self.labels[idx]))


def install_classifier_row_freeze(model: nn.Module):
    """Block gradient on every classifier row except the 6 targets."""
    non_target = torch.tensor(NON_TARGET, dtype=torch.long)
    def hook_w(grad):
        new_grad = grad.clone()
        new_grad[non_target] = 0
        return new_grad
    def hook_b(grad):
        new_grad = grad.clone()
        new_grad[non_target] = 0
        return new_grad
    model.dense.weight.register_hook(hook_w)
    model.dense.bias.register_hook(hook_b)


def evaluate(model, loader, device):
    model.eval()
    all_gt, all_pp = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="  val", leave=False):
            x = x.to(device)
            all_pp.append(torch.sigmoid(model(x)).cpu().numpy())
            all_gt.append(y.numpy())
    gt = np.concatenate(all_gt); pp = np.concatenate(all_pp)
    per_head = {}
    rocs, prs = [], []
    for idx in TARGET_IDX:
        pos = gt[:, idx].sum()
        if pos < 1 or pos == gt.shape[0]:
            per_head[idx] = dict(pos=int(pos), roc=float('nan'), pr=float('nan'))
            continue
        roc = float(roc_auc_score(gt[:, idx], pp[:, idx]))
        pr  = float(average_precision_score(gt[:, idx], pp[:, idx]))
        per_head[idx] = dict(pos=int(pos), roc=roc, pr=pr)
        rocs.append(roc); prs.append(pr)
    return per_head, float(np.mean(rocs)) if rocs else float('nan'), float(np.mean(prs)) if prs else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs',     type=int,   default=5)
    ap.add_argument('--batch-size', type=int,   default=64)
    ap.add_argument('--lr',         type=float, default=5e-4)  # higher OK — only 6,150 trainable params
    ap.add_argument('--weight-decay', type=float, default=1e-5)
    ap.add_argument('--device',     default=None)
    ap.add_argument('--seed',       type=int,   default=42)
    ap.add_argument('--pos-weight', action='store_true',
                    help="Use per-head pos_weight = n_neg/n_pos in BCEWithLogitsLoss "
                         "to counter PTB-XL class imbalance. Caps weight at 50 to "
                         "avoid destabilizing the SVT/VT heads (≈148 positives).")
    ap.add_argument('--pos-weight-cap', type=float, default=50.0)
    ap.add_argument('--out-suffix', default='v2',
                    help="Suffix on the output checkpoint filename, e.g. 'v2' or 'v3_posw'.")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = resolve_device(args.device)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    out_ckpt = Path(DEFAULT_OUT.format(suffix=args.out_suffix))
    print(f"Device: {device}")
    print(f"Target heads (scope-aligned): {TARGET_IDX} "
          f"({'idx 2=NormalECG, 4=SBR, 5=AFib, 6=STach, 93=SVT, 98=VT; Pause(142) excluded — 0 PTB-XL positives'})")
    print(f"Output checkpoint: {out_ckpt}")
    if args.pos_weight:
        print(f"pos_weight BCE: ENABLED (cap={args.pos_weight_cap})")

    print("\nBuilding datasets ...")
    train_ds = MultiAngleDataset('train', ANGLES_TRAIN)
    val_ds   = MultiAngleDataset('val',   ANGLES_VAL)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=128,             shuffle=False, num_workers=0)

    print("\nLoading base model and freezing backbone ...")
    model = load_ecgfounder(device)
    for p in model.parameters():
        p.requires_grad = False
    model.dense.weight.requires_grad = True
    model.dense.bias.requires_grad   = True
    install_classifier_row_freeze(model)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params (with row hook): {n_trainable} "
          f"(effective ~{6 * 1024 + 6} = 6,150 after row mask)")

    # Cache base classifier rows to verify they don't change post-training.
    base_w = model.dense.weight.detach().clone()
    base_b = model.dense.bias.detach().clone()

    # Build pos_weight tensor for the 6 target heads from the training labels.
    # Other 144 heads' pos_weight is 1.0 (irrelevant — masked out).
    pos_weight = torch.ones(150, device=device)
    if args.pos_weight:
        for idx in TARGET_IDX:
            n_pos = int(train_ds.labels[:, idx].sum())
            n_neg = len(train_ds.labels) - n_pos
            raw_w = n_neg / max(1, n_pos)
            capped = min(raw_w, args.pos_weight_cap)
            pos_weight[idx] = capped
            print(f"  pos_weight[{idx}] = {capped:.2f}  (raw n_neg/n_pos = {raw_w:.2f}, "
                  f"n_pos={n_pos}, n_neg={n_neg})")
    criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    target_mask = torch.zeros(150, device=device); target_mask[TARGET_IDX] = 1.0
    # weight_decay MUST be 0 here: even with the gradient hook zeroing rows,
    # Adam's wd term `θ_t = θ_t-1 - lr·(grad + wd·θ)` shrinks ALL rows each
    # step, including those whose gradient is zero. With wd=0 + the hook +
    # the manual restore below, non-target rows stay byte-identical to base.
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0,
    )
    if args.weight_decay != 0.0:
        print(f"  ⚠️ ignoring --weight-decay={args.weight_decay} — forced to 0 to "
              f"preserve non-target classifier rows.")

    # Pre-compute the non-target index buffer on-device for fast restore.
    non_target_t = torch.tensor(NON_TARGET, dtype=torch.long, device=device)
    # Cache base values for the non-target rows, on-device, for the restore step.
    base_w_non_target = base_w[non_target_t].clone()
    base_b_non_target = base_b[non_target_t].clone()

    print("\nBaseline val metrics:")
    per_head, macro_roc, macro_pr = evaluate(model, val_dl, device)
    for idx in TARGET_IDX:
        h = per_head[idx]
        print(f"  idx {idx:>3}: ROC={h['roc']:.4f}  PR={h['pr']:.4f}  pos={h['pos']}")
    print(f"  6-head macro: ROC={macro_roc:.4f}  PR={macro_pr:.4f}")
    best_macro_roc = macro_roc
    history = [{'epoch': 0, 'train_loss': None, 'macro_roc': macro_roc, 'macro_pr': macro_pr, 'per_head': per_head}]

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_start = time.time()
        running_loss, n_batches = 0.0, 0
        for x, y in tqdm(train_dl, desc=f"  epoch {epoch}/{args.epochs} train"):
            x = x.to(device); y = y.to(device)
            logits = model(x)
            per_elem = criterion(logits, y)
            # Mask loss to the 6 target heads. Sum over those heads, then mean over batch.
            loss = (per_elem * target_mask).sum() / (per_elem.shape[0] * target_mask.sum())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Belt-and-suspenders: manually restore non-target classifier rows.
            # The gradient hook ensures grad is 0 on those rows; weight_decay
            # is forced to 0 above; this restore catches any future regression
            # in either of those (e.g. an optimizer momentum quirk).
            with torch.no_grad():
                model.dense.weight.data[non_target_t] = base_w_non_target
                model.dense.bias.data[non_target_t]   = base_b_non_target

            running_loss += float(loss.item()); n_batches += 1
        train_loss = running_loss / max(1, n_batches)

        # Mid-training sanity check: non-target classifier rows must still equal base
        with torch.no_grad():
            w_diff = (model.dense.weight - base_w).abs()
            w_diff_non_target = w_diff[NON_TARGET].max().item()
            b_diff = (model.dense.bias - base_b).abs()
            b_diff_non_target = b_diff[NON_TARGET].max().item()
        assert w_diff_non_target < 1e-7, (
            f"Non-target classifier row drifted! Max delta = {w_diff_non_target}. "
            f"The gradient hook isn't working correctly."
        )

        per_head, macro_roc, macro_pr = evaluate(model, val_dl, device)
        print(f"  epoch {epoch}: train_loss={train_loss:.4f}  6-head macro ROC={macro_roc:.4f}  PR={macro_pr:.4f}  ({time.time()-ep_start:.1f}s)")
        print(f"    non-target row drift: weight_max={w_diff_non_target:.2e}, bias_max={b_diff_non_target:.2e}  ✓")
        for idx in TARGET_IDX:
            h = per_head[idx]
            print(f"      idx {idx:>3}: ROC={h['roc']:.4f}  PR={h['pr']:.4f}")

        history.append({
            'epoch': epoch, 'train_loss': train_loss,
            'macro_roc': macro_roc, 'macro_pr': macro_pr,
            'per_head': per_head,
            'non_target_w_drift': w_diff_non_target,
            'non_target_b_drift': b_diff_non_target,
        })

        if macro_roc > best_macro_roc:
            best_macro_roc = macro_roc
            out_ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_ckpt)
            print(f"    ✓ new best macro ROC — saved to {out_ckpt}")

    print(f"\nDone. Best 6-head macro ROC (val/fold9) = {best_macro_roc:.4f}")

    # ── Held-out TEST (fold 10) — evaluated ONCE on the model selected by val ──
    if out_ckpt.exists():
        print(f"\nReloading best checkpoint ({out_ckpt}) for held-out TEST ...")
        model.load_state_dict(torch.load(out_ckpt, map_location=device))
    else:
        print("\n⚠️ no checkpoint beat baseline val — testing the final-epoch model.")
    test_ds = MultiAngleDataset('test', ANGLES_VAL)        # fold 10
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)
    test_ph, test_roc, test_pr = evaluate(model, test_dl, device)
    print("\n" + "=" * 64)
    print("HELD-OUT TEST (PTB-XL fold 10) — scope-aligned 6-head metrics")
    print("=" * 64)
    NAMES = {2: 'NORMAL ECG', 4: 'Bradycardia', 5: 'AFib', 6: 'Sinus Tachy',
             93: 'SV Run(SVT)', 98: 'V Run(VT)'}
    for idx in TARGET_IDX:
        h = test_ph[idx]
        print(f"  idx {idx:>3} {NAMES.get(idx,''):<12}: ROC={h['roc']:.4f}  PR={h['pr']:.4f}  pos={h['pos']}")
    print(f"  6-head macro: ROC={test_roc:.4f}  PR={test_pr:.4f}")
    history.append({'epoch': 'TEST_fold10', 'macro_roc': test_roc,
                    'macro_pr': test_pr, 'per_head': test_ph})

    with open(LOG_DIR / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2, default=str)
    print(f"\nHistory → {LOG_DIR / 'training_history.json'}")


if __name__ == '__main__':
    main()
