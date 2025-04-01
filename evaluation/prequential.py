import numpy as np
import copy

from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.lazy import KNNClassifier, KNNADWINClassifier
from river.anomaly import OneClassSVM
from river import feature_extraction as fx

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from skmultiflow.drift_detection import ADWIN

import time

def run_prequential(classifier, stream, feature_selector=None, drift_detection=ADWIN(0.9), n_pretrain=200, preq_samples=100000):
    stream.restart()
    n_samples, correct_predictions = 0, 0
    true_labels, pred_labels = [], []
    processing_times = []
    drift_idx_list = []

    # pretrain samples
    for _ in range(n_pretrain):
        X_pretrain, y_pretrain = stream.next_sample()
        if isinstance(classifier, AdaptiveRandomForestClassifier) or isinstance(classifier, KNNClassifier) or isinstance(classifier, KNNADWINClassifier):
            classifier.partial_fit(X_pretrain, y_pretrain, classes=stream.target_values)
        else:
            classifier.learn_one(dict(enumerate(*X_pretrain)))
    
    # print(f"EVALUATION: Model pretrained on {n_pretrain} samples.")

    # prequential loop
    while n_samples < preq_samples and stream.has_more_samples():
        X, y = stream.next_sample()
        start = time.perf_counter()
        n_samples += 1

        # with dynamic feature selection
        if feature_selector is not None:
            feature_selector.weight_features(copy.copy(X), copy.copy(y))
            X_select = feature_selector.select_features(copy.copy(X), rng=np.random.default_rng())
            
            # Test first
            if isinstance(classifier, AdaptiveRandomForestClassifier) or isinstance(classifier, KNNClassifier) or isinstance(classifier, KNNADWINClassifier):
                y_pred = classifier.predict(X_select)
            else:
                score = classifier.score_one(dict(enumerate(*X_select)))
                y_pred = classifier.classify(score)
            
            # Train incrementally
            if isinstance(classifier, AdaptiveRandomForestClassifier) or isinstance(classifier, KNNClassifier) or isinstance(classifier, KNNADWINClassifier):
                classifier.partial_fit(copy.copy(X_select), [y[0]])
            else:
                classifier.learn_one(copy.copy(dict(enumerate(*X_select))))
        
        # no feature selection
        else:
            # Test first
            if isinstance(classifier, AdaptiveRandomForestClassifier) or isinstance(classifier, KNNClassifier) or isinstance(classifier, KNNADWINClassifier):
                y_pred = classifier.predict(X)
            else:
                score = classifier.score_one(dict(enumerate(*X)))
                y_pred = classifier.classify(score)
            
            # Train incrementally
            if isinstance(classifier, AdaptiveRandomForestClassifier) or isinstance(classifier, KNNClassifier) or isinstance(classifier, KNNADWINClassifier):
                classifier.partial_fit(copy.copy(X), [y[0]])
            else:
                classifier.learn_one(copy.copy(dict(enumerate(*X))))
        
        # drift detection
        if isinstance(drift_detection, ADWIN):
            if isinstance(classifier, AdaptiveRandomForestClassifier) or isinstance(classifier, KNNADWINClassifier):
                if classifier.drift_detection_method.detected_change():
                    drift_idx_list.append(n_samples - 1)
            else: # one class svm
                drift_detection.add_element(np.float64(y_pred == y))
                if drift_detection.detected_change():
                    drift_idx_list.append(n_samples - 1)
                    drift_detection.reset()
                    # reset one class svm model
                    classifier.anomaly_detector = (
                        fx.RBFSampler(gamma=classifier.anomaly_detector[0].gamma) | OneClassSVM(classifier.anomaly_detector[1].nu) # change with the optimized params
                    )

        # evaluation
        if y_pred == y:
            correct_predictions += 1
        true_labels.append(y[0])
        if isinstance(classifier, AdaptiveRandomForestClassifier) or isinstance(classifier, KNNClassifier) or isinstance(classifier, KNNADWINClassifier):
            pred_labels.append(y_pred[0])
        else:
            pred_labels.append(y_pred)
        
        end = time.perf_counter()
        processing_times.append(end - start)
    

    # evaluation metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    auc = roc_auc_score(true_labels, pred_labels)
    avg_processing_time = sum(processing_times) / len(processing_times)

    if feature_selector is None:
        return accuracy, precision, recall, f1, auc, avg_processing_time, drift_idx_list
    else:
        weights_history = feature_selector.weights_history
        selection_history = feature_selector.selected_features_history
        return accuracy, precision, recall, f1, auc, avg_processing_time, selection_history, weights_history, drift_idx_list