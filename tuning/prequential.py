import numpy as np
import copy
import skmultiflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import time
import tracemalloc
from tracemalloc import Snapshot


def run_prequential(classifier, stream, feature_selector, n_pretrain=200):

    stream.restart()
    n_samples, correct_predictions = 0, 0
    true_labels, pred_labels = [], []
    processing_times = []

    # print(f"Evaluating {setup_name} configuration.")

    # pretrain samples
    for _ in range(n_pretrain):
        X_pretrain, y_pretrain = stream.next_sample()
        # print(dict(enumerate(*X_pretrain)))
        if isinstance(classifier, skmultiflow.meta.AdaptiveRandomForestClassifier) or isinstance(classifier, skmultiflow.lazy.KNNClassifier) or isinstance(classifier, skmultiflow.lazy.KNNADWINClassifier):
            classifier.partial_fit(X_pretrain, y_pretrain, classes=stream.target_values)
        else:
            classifier.learn_one(dict(enumerate(*X_pretrain)))
    
    # print(f"Model pretrained on {n_pretrain} samples.")

    while n_samples < 50000 and stream.has_more_samples():
        X, y = stream.next_sample()
        start = time.perf_counter()
        n_samples += 1

        if feature_selector is not None:
            # with dynamic feature selection
            feature_selector.weight_features(copy.copy(X), copy.copy(y))
            X_select = feature_selector.select_features(copy.copy(X), rng=np.random.default_rng())
            # y_pred = classifier.predict(X_select)
            if isinstance(classifier, skmultiflow.meta.AdaptiveRandomForestClassifier) or isinstance(classifier, skmultiflow.lazy.KNNClassifier) or isinstance(classifier, skmultiflow.lazy.KNNADWINClassifier):
                y_pred = classifier.predict(X_select)
            else:
                score = classifier.score_one(dict(enumerate(*X_select)))
                y_pred = classifier.classify(score)
            
            # Train incrementally
            if isinstance(classifier, skmultiflow.meta.AdaptiveRandomForestClassifier) or isinstance(classifier, skmultiflow.lazy.KNNClassifier) or isinstance(classifier, skmultiflow.lazy.KNNADWINClassifier):
                classifier.partial_fit(copy.copy(X_select), [y[0]])
            else:
                classifier.learn_one(copy.copy(dict(enumerate(*X_select))))

        else:
            # no feature selection
            if isinstance(classifier, skmultiflow.meta.AdaptiveRandomForestClassifier) or isinstance(classifier, skmultiflow.lazy.KNNClassifier) or isinstance(classifier, skmultiflow.lazy.KNNADWINClassifier):
                y_pred = classifier.predict(X)
            else:
                score = classifier.score_one(dict(enumerate(*X)))
                y_pred = classifier.classify(score)
            
            # Train incrementally
            if isinstance(classifier, skmultiflow.meta.AdaptiveRandomForestClassifier) or isinstance(classifier, skmultiflow.lazy.KNNClassifier) or isinstance(classifier, skmultiflow.lazy.KNNADWINClassifier):
                classifier.partial_fit(copy.copy(X), [y[0]])
            else:
                classifier.learn_one(copy.copy(dict(enumerate(*X))))
        
        if y_pred == y:
            correct_predictions += 1
        
        true_labels.append(y[0])
        if isinstance(classifier, skmultiflow.meta.AdaptiveRandomForestClassifier) or isinstance(classifier, skmultiflow.lazy.KNNClassifier) or isinstance(classifier, skmultiflow.lazy.KNNADWINClassifier):
            pred_labels.append(y_pred[0])
        else:
            pred_labels.append(y_pred)

        # check for drift
        # if drift_detector is not None:
        #     drift_detector.add_element(np.float64(y_pred == y))
        #     if drift_detector.detected_change():
        #         print(f"drift detected at {n_samples}")
        end = time.perf_counter()
        processing_times.append(end - start)


    # Calculate accuracy
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    avg_processing_time = sum(processing_times) / len(processing_times)

    return accuracy, precision, recall, f1, avg_processing_time