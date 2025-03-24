import numpy as np
import copy
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.lazy import KNNClassifier, KNNADWINClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skmultiflow.drift_detection import ADWIN
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

import time

def run_prequential(classifier, stream, feature_selector=None, drift_detector=ADWIN(), n_pretrain=200, preq_samples=50000):
    stream.restart()
    n_samples, correct_predictions = 0, 0
    true_labels, pred_labels = [], []
    processing_times = []
    pred_probabilities = [] # Added to store predicted probabilities

    # uncomment after tuning
    # print(f"Evaluating {setup_name} configuration.")

    # pretrain samples
    # for _ in range(n_pretrain):
    #     X_pretrain, y_pretrain = stream.next_sample()
    #     # print(dict(enumerate(*X_pretrain)))
    #     if isinstance(classifier, AdaptiveRandomForestClassifier) or isinstance(classifier, KNNClassifier) or isinstance(classifier, KNNADWINClassifier):
    #         classifier.partial_fit(X_pretrain, y_pretrain, classes=stream.target_values)
    #     else:
    #         classifier.learn_one(dict(enumerate(*X_pretrain)))

    # print(f"Model pretrained on {n_pretrain} samples.")

    while n_samples < preq_samples and stream.has_more_samples():
        X, y = stream.next_sample()
        start = time.perf_counter()
        n_samples += 1

        if feature_selector is not None:
            # with dynamic feature selection
            feature_selector.weight_features(copy.copy(X), copy.copy(y))
            X_select = feature_selector.select_features(copy.copy(X), rng=np.random.default_rng())

            # Test first
            if isinstance(classifier, AdaptiveRandomForestClassifier) or isinstance(classifier, KNNClassifier) or isinstance(classifier, KNNADWINClassifier):
                y_pred = classifier.predict(X_select)
                y_prob = classifier.predict_proba(X_select) #For AUC
            else:
                score = classifier.score_one(dict(enumerate(*X_select)))
                y_pred = classifier.classify(score)
                y_prob = classifier.predict_proba_one(dict(enumerate(*X_select))) #For AUC

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
                y_prob = classifier.predict_proba(X) #For AUC
            else:
                score = classifier.score_one(dict(enumerate(*X)))
                y_pred = classifier.classify(score)
                y_prob = classifier.predict_proba_one(dict(enumerate(*X))) #For AUC

            # Train incrementally
            if isinstance(classifier, AdaptiveRandomForestClassifier) or isinstance(classifier, KNNClassifier) or isinstance(classifier, KNNADWINClassifier):
                classifier.partial_fit(copy.copy(X), [y[0]])
            else:
                classifier.learn_one(copy.copy(dict(enumerate(*X))))

        if y_pred == y:
            correct_predictions += 1
        true_labels.append(y[0])
        if isinstance(classifier, AdaptiveRandomForestClassifier) or isinstance(classifier, KNNClassifier) or isinstance(classifier, KNNADWINClassifier):
            pred_labels.append(y_pred[0])
            pred_probabilities.append(y_prob[0][1]) # Assuming binary classification, probability of class 1
        else:
            pred_labels.append(y_pred)
            pred_probabilities.append(y_prob[1]) # Assuming binary classification, probability of class 1

        # check for drift
        # if drift_detector is not None:
        #     drift_detector.add_element(np.float64(y_pred == y))
        #     if drift_detector.detected_change():
        #         print(f"drift detected at {n_samples}")
        #         # classifier.reset()

        end = time.perf_counter()
        processing_times.append(end - start)

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    avg_processing_time = sum(processing_times) / len(processing_times)
    auc = roc_auc_score(true_labels, pred_labels)

    return accuracy, precision, recall, f1, auc, avg_processing_time