"""Fine-tune the 1-lead ECGFounder on the fuzzylead2 dataset.

Train set : concatenation of fuzzylead2/ptbxl/train_{45,75,90}deg.npz (58,803 records)
Val set   : fuzzylead2/ptbxl/val_75deg.npz (2,198 records — Lead II proxy)
Loss      : BCE-with-logits (multi-label, 150 heads)
Optimizer : Adam(lr=1e-4, weight_decay=1e-5)
Epochs    : 5  (saves the best-ROC checkpoint to checkpoint/1_lead_ECGFounder_fuzzy.pth)
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from checkpoints import load_ecgfounder
from device_utils import resolve_device

DATA_DIR     = Path("data/fuzzylead2/ptbxl")
TRAIN_ANGLES = (45, 75, 90)
VAL_ANGLE    = 75
DEFAULT_CKPT = Path("checkpoint/1_lead_ECGFounder_fuzzy.pth")
LOG_DIR      = Path("res/finetune_fuzzylead2")


class FuzzylNpzDataset(Dataset):
    """Loads one or more *.npz files (ecg, labels) and concatenates them.

    If `split` is given ('train'|'val'|'test'), rows are filtered by ecg_id to the
    recommended PTB-XL fold convention (ptbxl_splits): train=folds1-8, val=fold9,
    test=fold10 — no patient crosses the boundary. Requires ecg_ids in the npz.
    """
    def __init__(self, files, split=None):
        ecg_list, lbl_list = [], []
        for f in files:
            z = np.load(f)
            ecg, lbl = z['ecg'].astype(np.float32), z['labels'].astype(np.float32)
            if split is not None:
                if 'ecg_ids' not in z.files:
                    raise ValueError(f"{f} has no ecg_ids — cannot apply fold split")
                import ptbxl_splits
                keep = ptbxl_splits.mask_for(z['ecg_ids'], split)
                ecg, lbl = ecg[keep], lbl[keep]
            ecg_list.append(ecg); lbl_list.append(lbl)
        self.ecg    = np.concatenate(ecg_list, axis=0)
        self.labels = np.concatenate(lbl_list, axis=0)
        assert self.ecg.shape[0] == self.labels.shape[0]
        assert self.ecg.shape[1] == 1 and self.ecg.shape[2] == 5000
        assert self.labels.shape[1] == 150

    def __len__(self):
        return self.ecg.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.ecg[idx]), torch.from_numpy(self.labels[idx])


def evaluate(model, loader, device):
    model.eval()
    all_gt, all_pp = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="  val", leave=False):
            x = x.to(device)
            logits = model(x)
            all_pp.append(torch.sigmoid(logits).cpu().numpy())
            all_gt.append(y.numpy())
    gt = np.concatenate(all_gt)
    pp = np.concatenate(all_pp)

    # Per-class ROC for classes with at least one positive.
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    rocs, prs = [], []
    for c in range(gt.shape[1]):
        pos = gt[:, c].sum()
        if pos < 1 or pos == gt.shape[0]:
            continue
        rocs.append(roc_auc_score(gt[:, c], pp[:, c]))
        prs.append(average_precision_score(gt[:, c], pp[:, c]))
    macro_roc = float(np.mean(rocs)) if rocs else float('nan')
    macro_pr  = float(np.mean(prs))  if prs  else float('nan')
    # Micro-averaged F1 at 0.5
    pred_bin = (pp >= 0.5).astype(int)
    micro_f1 = float(f1_score(gt.flatten(), pred_bin.flatten(), zero_division=0))
    return macro_roc, macro_pr, micro_f1, len(rocs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs',     type=int,   default=5)
    ap.add_argument('--batch-size', type=int,   default=32)
    ap.add_argument('--lr',         type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-5)
    ap.add_argument('--workers',    type=int,   default=0)
    ap.add_argument('--device',     default=None)
    ap.add_argument('--seed',       type=int,   default=42)
    ap.add_argument('--masked-loss', action='store_true',
                    help="Compute BCE loss only over heads with positives in the "
                         "training data; gradients for inactive heads are zero. "
                         "Preserves base-model behavior on unrepresented classes.")
    ap.add_argument('--out-ckpt',   default=None,
                    help="Override output checkpoint path (defaults to "
                         "1_lead_ECGFounder_fuzzy.pth, or _masked.pth with --masked-loss).")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = resolve_device(args.device)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    # Recommended PTB-XL convention: train=folds1-8, val=fold9 — BOTH from the
    # train_*deg.npz (folds 1-9), masked by ecg_id. Fold 10 (val_*deg.npz) is the
    # held-out TEST set and is intentionally NOT used during training.
    print(f"Loading train (folds 1-8): {TRAIN_ANGLES}-deg combined ...")
    train_files = [DATA_DIR / f"train_{a}deg.npz" for a in TRAIN_ANGLES]
    train_ds = FuzzylNpzDataset(train_files, split='train')
    print(f"  train n = {len(train_ds)}")
    val_ds = FuzzylNpzDataset(train_files, split='val')           # fold 9
    print(f"  val   n = {len(val_ds)}  (fold 9; fold 10 reserved as test)")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  num_workers=args.workers, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=64,
                          shuffle=False, num_workers=args.workers, pin_memory=False)

    # Resolve checkpoint output path
    out_ckpt = Path(args.out_ckpt) if args.out_ckpt else (
        Path("checkpoint/1_lead_ECGFounder_fuzzy_masked.pth")
        if args.masked_loss else DEFAULT_CKPT
    )

    # Compute per-head loss mask (1.0 for heads with ≥1 positive in train)
    class_pos = train_ds.labels.sum(axis=0)
    active_mask = (class_pos > 0).astype(np.float32)
    active_idx  = np.where(active_mask > 0)[0]
    print(f"Active classes in training data ({int(active_mask.sum())}/150): "
          f"{active_idx.tolist()}")
    if args.masked_loss:
        print(f"Masked-loss mode: gradient is zero for {150 - int(active_mask.sum())} inactive heads.")
    loss_mask = torch.from_numpy(active_mask).to(device)

    print(f"Loading pretrained 1-lead ECGFounder ...")
    model = load_ecgfounder(device)
    for p in model.parameters():
        p.requires_grad = True
    model.train()

    # Per-element BCE so we can apply the mask before reduction
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"\nBaseline (pretrained) val metrics:")
    roc, pr, f1, k = evaluate(model, val_dl, device)
    print(f"  macro ROC = {roc:.4f}   macro PR = {pr:.4f}   micro F1 = {f1:.4f}   active classes = {k}")
    best_roc = roc
    history = [{'epoch': 0, 'train_loss': None, 'val_macro_roc': roc, 'val_macro_pr': pr, 'val_micro_f1': f1}]

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_start = time.time()
        running_loss, n_batches = 0.0, 0
        for x, y in tqdm(train_dl, desc=f"  epoch {epoch}/{args.epochs} train"):
            x = x.to(device); y = y.to(device)
            logits = model(x)
            per_elem = criterion(logits, y)  # (batch, 150)
            if args.masked_loss:
                # Zero out the 140 heads with no training signal, then
                # normalize by the active-head count so the loss magnitude
                # is comparable to the unmasked version.
                per_elem = per_elem * loss_mask
                loss = per_elem.sum() / (per_elem.shape[0] * loss_mask.sum())
            else:
                loss = per_elem.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()); n_batches += 1
        train_loss = running_loss / max(1, n_batches)

        roc, pr, f1, k = evaluate(model, val_dl, device)
        print(f"  epoch {epoch}: train_loss={train_loss:.4f}  val macro ROC={roc:.4f}  PR={pr:.4f}  F1={f1:.4f}  ({time.time()-ep_start:.1f}s)")
        history.append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_macro_roc': roc, 'val_macro_pr': pr, 'val_micro_f1': f1
        })

        if roc > best_roc:
            best_roc = roc
            out_ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_ckpt)
            print(f"    ✓ new best ROC — saved to {out_ckpt}")

    print(f"\nDone. Best val macro ROC = {best_roc:.4f}")
    hist_name = "training_history_masked.json" if args.masked_loss else "training_history.json"
    with open(LOG_DIR / hist_name, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"History written to {LOG_DIR / hist_name}")


if __name__ == '__main__':
    main()
