from prequential import run_prequential

from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.drift_detection.adwin import ADWIN

from float.feature_selection import FIRES, OFS           # online feature methods
from skmultiflow.data import FileStream             # create stream from file

import pandas as pd
import csv

data_loader = FileStream(filepath='merged_cesnet.csv')

ref_sample, _ = data_loader.next_sample(50)
data_loader.reset()

arf_params = {
    'n_estimators': 6,
    'max_features': "auto",
    'drift_detection_method': None,
    'warning_detection_method': None,
    'grace_period': 25,
    'split_criterion': "gini",
    'split_confidence': 0.01,
    'tie_threshold': 0.01,
    'leaf_prediction': "nba"
}

fires_params = {
    'penalty_s': 0.1,
    'penalty_r': 1,
    'lr_mu': 0.025,
    'lr_sigma': 0.1,
    'n_total_features': data_loader.n_features,
    'n_selected_features': 10,
    'classes': data_loader.target_values,
    'baseline': "gaussian",
    'ref_sample': ref_sample,
    'reset_after_drift': True
}

new_arf_params = arf_params.copy()
new_arf_params.update({'max_features': None, 'drift_detection_method': ADWIN(0.9), 'warning_detection_method': ADWIN(0.7)})

# FIRES
print("EVALUATING: ARF + ADWIN + FIRES")
accuracy, precision, recall, f1, auc, fpr, tpr, avg_processing_time, mem_usage, selection_history, weights_history, drift_idx_list = run_prequential(
    classifier=AdaptiveRandomForestClassifier(**new_arf_params),
    stream=data_loader, 
    feature_selector=FIRES(**fires_params), 
    drift_detection=ADWIN(), 
    preq_samples=data_loader.n_remaining_samples()
)
print(f"Accuracy: {accuracy:.6f}, Precision: {precision:.6f}, Recall: {recall:.6f}, F1: {f1:.6f}, AUC: {auc:.6f}")
print(f"Average processing time: {avg_processing_time:.7f} s")
print(f"Memory used: {mem_usage:.6f} MB")
print(f"FPR: {list(fpr)}")
print(f"TPR: {list(tpr)}")
print(drift_idx_list)
print()

with open('sel_features_fires.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(selection_history)
with open('weights_fires.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(weights_history)