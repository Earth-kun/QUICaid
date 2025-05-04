import numpy as np
import copy

from skmultiflow.meta import AdaptiveRandomForestClassifier
# from skmultiflow.lazy import KNNClassifier, KNNADWINClassifier
# from river.anomaly import OneClassSVM
# from river import feature_extraction as fx

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from skmultiflow.drift_detection import ADWIN

# from timeit import default_timer as timer
import warnings

import argparse
from threading import Thread, Event

# Argument Parsing
parser = argparse.ArgumentParser(description="Evaluate flows from FIFO file.")
parser.add_argument("--n_estimators", type=int, default=6, help="Lorem Ipsum.")
parser.add_argument("--max_features", type=str, default="auto", help="Lorem Ipsum.")
parser.add_argument("--grace_period", type=int, default=25, help="Lorem Ipsum.")
parser.add_argument("--split_criterion", type=str, default="gini", help="Lorem Ipsum.")
parser.add_argument("--split_confidence", type=float, default=0.01, help="Lorem Ipsum.")
parser.add_argument("--tie_threshold", type=float, default=0.01, help="Lorem Ipsum.")
parser.add_argument("--leaf_prediction", type=str, default="nba", help="Lorem Ipsum.")
# parser.add_argument("--output", type=str, default=None, help="Lorem Ipsum.")
args = parser.parse_args()

FIFO_FILE = "/tmp/flow_fifo"
N_ESTIMATORS = args.n_estimators
MAX_FEATURES = args.max_features
GRACE_PERIOD = args.grace_period
SPLIT_CRITERION = args.split_criterion
SPLIT_CONFIDENCE = args.split_confidence
TIE_THRESHOLD = args.tie_threshold
LEAF_PREDICTION = args.leaf_prediction
# OUTPUT_CSV = args.output

arf_params = {
    'n_estimators': N_ESTIMATORS,
    'max_features': MAX_FEATURES,
    'drift_detection_method': ADWIN(0.9),
    'warning_detection_method': ADWIN(0.7),
    'grace_period': GRACE_PERIOD,
    'split_criterion': SPLIT_CRITERION,
    'split_confidence': SPLIT_CONFIDENCE,
    'tie_threshold': TIE_THRESHOLD,
    'leaf_prediction': LEAF_PREDICTION,
}

classifier = AdaptiveRandomForestClassifier(**arf_params)

def read_fifo():
    """Reads packets from FIFO and batches them for processing."""
    global flow

    with open(FIFO_FILE, "r") as fifo:
        for line in fifo:
            flow = line.strip().split("\t")

            if stop_event.is_set():
                break

def run_prequential(classifier, stream, feature_selector=None, drift_detection=ADWIN(0.9), n_pretrain=200, preq_samples=100000):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    stream.restart()
    n_samples = 0
    true_labels, pred_labels = [], []
    # processing_times = []
    # drift_idx_list = []
    # pred_probabilities = []

    # # pretraining phase
    # if isinstance(classifier, AdaptiveRandomForestClassifier) or isinstance(classifier, KNNClassifier) or isinstance(classifier, KNNADWINClassifier):
    #     # for _ in range(9900):
    #     #     stream.next_sample()
    #     stream.next_sample(9900)
    
    #     for _ in range(n_pretrain):
    #         X_pretrain, y_pretrain = stream.next_sample()
    #         classifier.partial_fit(X_pretrain, y_pretrain, classes=stream.target_values)
        
    #     if isinstance(classifier, AdaptiveRandomForestClassifier):
    #         prev_drift_counts = [learner.nb_drifts_detected for learner in classifier.ensemble]
    # else:
    #     for _ in range(n_pretrain):
    #         X_pretrain, _ = stream.next_sample()
    #         classifier.learn_one(dict(enumerate(*X_pretrain)))
        
    #     stream.next_sample(9900)
        

    # prequential loop
    while n_samples < preq_samples and stream.has_more_samples():
        X, y = stream.next_sample()
        try:
            if X is not None and y is not None:
                n_samples += 1

                # Test first
                if isinstance(classifier, AdaptiveRandomForestClassifier):
                    y_pred = classifier.predict(X)
                
                # Train incrementally
                if isinstance(classifier, AdaptiveRandomForestClassifier):
                    classifier.partial_fit(copy.copy(X), [y[0]])

                # evaluation
                true_labels.append(y[0])
                if isinstance(classifier, AdaptiveRandomForestClassifier):
                    pred_labels.append(y_pred[0])
                

        except BaseException as e:
            print(e)
            break

    # evaluation metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)

    return accuracy, precision, recall, f1