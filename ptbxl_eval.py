import numpy as np
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os
import json
from device_utils import resolve_device
from util import eval_with_dynamic_thresh
from preprocessing import ECGPreprocessor
from checkpoints import load_ecgfounder
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

class PTBXL_SingleLead_Dataset(torch.utils.data.Dataset):
    """PTB-XL evaluation dataset.

    Uses the unified `ECGPreprocessor.for_ptbxl()` pipeline (50 Hz notch +
    0.67–40 Hz bandpass + median baseline + winsorized z-score).

    NOTE: This applies full filtering (previously PTB-XL eval skipped it),
    bringing PTB-XL inputs onto the same preprocessing manifold as fzark
    and MIT-BIH. Metrics will shift relative to the historical baseline.
    """

    def __init__(self, ecg_path, csv_path):
        self.data = pd.read_csv(csv_path)
        self.data = self.data.dropna(subset=['filename_hr', 'label'])
        self.ecg_path = ecg_path
        self.prep = ECGPreprocessor.for_ptbxl()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        hash_file_name = row['filename_hr']
        label = json.loads(row['label'])
        label = torch.tensor(label, dtype=torch.float)

        record_path = os.path.join(self.ecg_path, hash_file_name)
        tensor = self.prep.from_wfdb(record_path)
        return tensor, label
    
saved_dir = './res/eval_lead_ii'
os.makedirs(saved_dir, exist_ok=True)
csv_filepath = './csv/ptbxl_val.csv'
ecg_filepath = './data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
tasks = []
batch_size = 512
with open(os.path.join('./tasks.txt'), 'r') as fin:
    for line in fin:
        tasks.append(line.strip())

testset = PTBXL_SingleLead_Dataset(ecg_path=ecg_filepath, csv_path=csv_filepath)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
device = resolve_device()        # project default: MPS → CUDA → CPU

# Load model (auto-downloads checkpoint if missing).
model = load_ecgfounder(device)
for param in model.parameters():
    param.requires_grad = False
prog_iter_test = tqdm(testloader, desc="Testing", leave=False)
all_gt = []
all_pred_prob = []
all_thre_df = []

with torch.no_grad():
    for batch_idx, batch in enumerate(prog_iter_test):
        input_x, input_y = tuple(t.to(device) for t in batch)
        logits = model(input_x)
        pred = F.sigmoid(logits)
        all_pred_prob.append(pred.cpu().data.numpy())
        all_gt.append(input_y.cpu().data.numpy())
all_pred_prob = np.concatenate(all_pred_prob)
all_gt = np.concatenate(all_gt)
df_gt = pd.DataFrame(all_gt)
df_gt.to_csv(os.path.join(saved_dir, 'all_gt.csv'), index=False, float_format='%.5f')
df_pred = pd.DataFrame(all_pred_prob)
df_pred.to_csv(os.path.join(saved_dir, 'all_pred.csv'), index=False, float_format='%.5f')
res_test, res_test_auroc, res_test_sens, res_test_spec, res_test_f1, optimal_thresholds = eval_with_dynamic_thresh(all_gt, all_pred_prob)

for i, task in enumerate(tasks):
  all_thre_df.append([task, res_test_auroc[i], res_test_sens[i], res_test_spec[i],res_test_f1[i],optimal_thresholds[i]])

columns = ['Field_ID', 'AUROC', 'sensitivity', 'specificity', 'f1', 'optimal_thresholds']
all_thre_df = pd.DataFrame(all_thre_df, columns=columns)
all_thre_df.to_csv(os.path.join(saved_dir, 'res_thre.csv'), index=False, float_format='%.5f')

df_gt = pd.read_csv(os.path.join(saved_dir, 'all_gt.csv'))
df_pred = pd.read_csv(os.path.join(saved_dir, 'all_pred.csv'))

all_gt_df = df_gt.T
all_pred_df = df_pred.T

all_thre_df = pd.read_csv(os.path.join(saved_dir, 'res_thre.csv'))
all_thre_df = all_thre_df.iloc[:, -1].T

all_pred_df.index = all_pred_df.index.astype(int)
all_gt_df.index = all_gt_df.index.astype(int)

def calculate_performance_metrics(true, pred, threshold):
    true = np.array(true) 
    pred = np.array(pred) 
    pred_binary = (pred >= threshold).astype(int)  
    
    true_positive = np.sum((true == 1) & (pred_binary == 1))
    false_positive = np.sum((true == 0) & (pred_binary == 1))
    true_negative = np.sum((true == 0) & (pred_binary == 0))
    false_negative = np.sum((true == 1) & (pred_binary == 0))
    
    sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    ppv = precision
    npv = true_negative / (true_negative + false_negative) if (true_negative + false_negative) > 0 else 0
    
    if (precision + sensitivity) > 0:
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    else:
        f1_score = 0
    
    if len(np.unique(true)) == 1: 
        auroc = np.nan
        auprc = np.nan
    else:
        auroc = roc_auc_score(true, pred)
        auprc = average_precision_score(true, pred)
    
    return sensitivity, specificity, precision, f1_score, ppv, npv, auroc, auprc

def bootstrap_ci(metric_func, true, pred, threshold, n_resamples=10):
    true = np.array(true)  
    pred = np.array(pred)  
    pred_binary = (pred >= threshold).astype(int)  
    
    bootstrap_distribution = []
    for _ in range(n_resamples):
        indices = np.random.choice(len(true), len(true), replace=True)
        resampled_true = true[indices]
        resampled_pred = pred[indices]
        
        metric_value = metric_func(resampled_true, resampled_pred, threshold)
        bootstrap_distribution.append(metric_value)
    
    lower_bound = np.percentile(bootstrap_distribution, 2.5)
    upper_bound = np.percentile(bootstrap_distribution, 97.5)
    
    return (round(lower_bound, 3), round(upper_bound, 3))

results = []
for i, task in enumerate(tasks):
    true = all_gt_df.loc[i]
    pred = all_pred_df.loc[i]
    threshold = all_thre_df.loc[i]
    sens, spec, prec, f1, ppv, npv, auroc, auprc = calculate_performance_metrics(true, pred, threshold)
    
    sens_ci = bootstrap_ci(lambda true, pred, threshold: calculate_performance_metrics(true, pred, threshold)[0], true, pred, threshold)
    spec_ci = bootstrap_ci(lambda true, pred, threshold: calculate_performance_metrics(true, pred, threshold)[1], true, pred, threshold)
    f1_ci = bootstrap_ci(lambda true, pred, threshold: calculate_performance_metrics(true, pred, threshold)[3], true, pred, threshold)
    ppv_ci = bootstrap_ci(lambda true, pred, threshold: calculate_performance_metrics(true, pred, threshold)[4], true, pred, threshold)
    npv_ci = bootstrap_ci(lambda true, pred, threshold: calculate_performance_metrics(true, pred, threshold)[5], true, pred, threshold)
    auroc_ci = bootstrap_ci(lambda true, pred, threshold: calculate_performance_metrics(true, pred, threshold)[6], true, pred, threshold)
    auprc_ci = bootstrap_ci(lambda true, pred, threshold: calculate_performance_metrics(true, pred, threshold)[7], true, pred, threshold)
    
    results.append({
        'Label': task,
        'Sensitivity': round(sens, 3), 'Sensitivity_CI': sens_ci,
        'Specificity': round(spec, 3), 'Specificity_CI': spec_ci,
        'F1': round(f1, 3), 'F1_CI': f1_ci,
        'PPV': round(ppv, 3), 'PPV_CI': ppv_ci,
        'NPV': round(npv, 3), 'NPV_CI': npv_ci,
        'AUROC': round(auroc, 3) if not np.isnan(auroc) else np.nan, 'AUROC_CI': auroc_ci,
        'AUPRC': round(auprc, 3) if not np.isnan(auprc) else np.nan, 'AUPRC_CI': auprc_ci
    })

results_df = pd.DataFrame(results)
results_df.to_csv((os.path.join(saved_dir, 'res.csv')), index=False)
print(f"Validation evaluation finished successfully! Performance metrics are saved at {os.path.join(saved_dir, 'res.csv')}")
