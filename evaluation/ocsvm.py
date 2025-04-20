from prequential import run_prequential

from river import anomaly
from river.anomaly import OneClassSVM
from river import feature_extraction as fx
from skmultiflow.drift_detection.adwin import ADWIN

from float.feature_selection import FIRES, OFS           # online feature methods
from skmultiflow.data import FileStream             # create stream from file

import pandas as pd
import csv

data_loader = FileStream(filepath='merged_cesnet.csv')

ref_sample, _ = data_loader.next_sample(50)
data_loader.reset()

svm_params = {
    'q': 0.8,
    'gamma': 1,
    'nu': 0.05
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
    'ref_sample': ref_sample
}

ofs_params = {
    'n_selected_features': 5,
    'n_total_features': data_loader.n_features,
    'baseline': "gaussian",
    'ref_sample': ref_sample    
}

data_loader.restart()

########## WITHOUT ADWIN #########

# no feature selection
print("EVALUATING: OC-SVM")
accuracy, precision, recall, f1, auc, _, _, avg_processing_time, _ = run_prequential(
    classifier=anomaly.QuantileFilter(
        (fx.RBFSampler(gamma=svm_params['gamma']) | OneClassSVM(nu=svm_params['nu'])),
        q=svm_params['q']
    ), 
    stream=data_loader, 
    feature_selector=None, 
    drift_detection=None, 
    preq_samples=15000
)

print(f"Accuracy: {accuracy:.6f}, Precision: {precision:.6f}, Recall: {recall:.6f}, F1: {f1:.6f}, AUC: {auc:.6f}")
print(f"Average processing time: {avg_processing_time:.7f} s")
print(f"Memory used: {mem_usage:.6f} MB")
print()

data_loader.restart()

# FIRES

print("EVALUATING: OC-SVM + FIRES")
accuracy, precision, recall, f1, auc, _, _, avg_processing_time, mem_usage, _, _, _ = run_prequential(
    classifier=anomaly.QuantileFilter(
        (fx.RBFSampler(gamma=svm_params['gamma']) | OneClassSVM(nu=svm_params['nu'])),
        q=svm_params['q']
    ), 
    stream=data_loader, 
    feature_selector=FIRES(**fires_params), 
    drift_detection=None, 
    preq_samples=data_loader.n_remaining_samples()
)

print(f"Accuracy: {accuracy:.6f}, Precision: {precision:.6f}, Recall: {recall:.6f}, F1: {f1:.6f}, AUC: {auc:.6f}")
print(f"Average processing time: {avg_processing_time:.7f} s")
print(f"Memory used: {mem_usage:.6f} MB")
print()

data_loader.restart()

# OFS
print("EVALUATING: OC-SVM + OFS")
accuracy, precision, recall, f1, auc, _, _, avg_processing_time, mem_usage, _, _, _ = run_prequential(
    classifier=anomaly.QuantileFilter(
        (fx.RBFSampler(gamma=svm_params['gamma']) | OneClassSVM(nu=svm_params['nu'])),
        q=svm_params['q']
    ),   
    stream=data_loader, 
    feature_selector=OFS(**ofs_params), 
    drift_detection=None, 
    preq_samples=data_loader.n_remaining_samples()
)
print(f"Accuracy: {accuracy:.6f}, Precision: {precision:.6f}, Recall: {recall:.6f}, F1: {f1:.6f}, AUC: {auc:.6f}")
print(f"Average processing time: {avg_processing_time:.7f} s")
print(f"Memory used: {mem_usage:.6f} MB")
print()

data_loader.restart()


########## WITH ADWIN ##########


# no feature selection
print("EVALUATING: OC-SVM + ADWIN")
accuracy, precision, recall, f1, auc, fpr, tpr, avg_processing_time, mem_usage, drift_idx_list = run_prequential(
    classifier=anomaly.QuantileFilter(
        (fx.RBFSampler(gamma=svm_params['gamma']) | OneClassSVM(nu=svm_params['nu'])),
        q=svm_params['q']
    ),  
    stream=data_loader, 
    feature_selector=None, 
    drift_detection=ADWIN(0.001), 
    preq_samples=data_loader.n_remaining_samples()
)
print(f"Accuracy: {accuracy:.6f}, Precision: {precision:.6f}, Recall: {recall:.6f}, F1: {f1:.6f}, AUC: {auc:.6f}")
print(f"Average processing time: {avg_processing_time:.7f} s")
print(f"Memory used: {mem_usage:.6f} MB")
print(f"FPR: {list(fpr)}")
print(f"TPR: {list(tpr)}")
print(drift_idx_list)
print()

data_loader.restart()


# FIRES
print("EVALUATING: OC-SVM + ADWIN + FIRES")
accuracy, precision, recall, f1, auc, fpr, tpr, avg_processing_time, mem_usage, _, _, drift_idx_list = run_prequential(
    classifier=anomaly.QuantileFilter(
        (fx.RBFSampler(gamma=svm_params['gamma']) | OneClassSVM(nu=svm_params['nu'])),
        q=svm_params['q']
    ), 
    stream=data_loader, 
    feature_selector=FIRES(**fires_params), 
    drift_detection=ADWIN(0.001), 
    preq_samples=data_loader.n_remaining_samples()
)
print(f"Accuracy: {accuracy:.6f}, Precision: {precision:.6f}, Recall: {recall:.6f}, F1: {f1:.6f}, AUC: {auc:.6f}")
print(f"Average processing time: {avg_processing_time:.7f} s")
print(f"Memory used: {mem_usage:.6f} MB")
print(f"FPR: {list(fpr)}")
print(f"TPR: {list(tpr)}")
print(drift_idx_list)
print()

data_loader.restart()    

# OFS
print("EVALUATING: ARF + ADWIN + OFS")
accuracy, precision, recall, f1, auc, fpr, tpr, avg_processing_time, mem_usage, selection_history, weights_history, drift_idx_list = run_prequential(
    classifier=anomaly.QuantileFilter(
        (fx.RBFSampler(gamma=svm_params['gamma']) | OneClassSVM(nu=svm_params['nu'])),
        q=svm_params['q']
    ), 
    stream=data_loader, 
    feature_selector=OFS(**ofs_params), 
    drift_detection=ADWIN(0.001), 
    preq_samples=data_loader.n_remaining_samples()
)
print(f"Accuracy: {accuracy:.6f}, Precision: {precision:.6f}, Recall: {recall:.6f}, F1: {f1:.6f}, AUC: {auc:.6f}")
print(f"Average processing time: {avg_processing_time:.7f} s")
print(f"Memory used: {mem_usage:.6f} MB")
print(f"FPR: {list(fpr)}")
print(f"TPR: {list(tpr)}")
print(drift_idx_list)
print()