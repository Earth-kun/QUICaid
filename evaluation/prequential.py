import numpy as np
import copy

from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.lazy import KNNClassifier, KNNADWINClassifier
from river.anomaly import OneClassSVM
from river import feature_extraction as fx

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from skmultiflow.drift_detection import ADWIN

from timeit import default_timer as timer
import warnings

def run_prequential(classifier, stream, feature_selector=None, drift_detection=ADWIN(0.9), n_pretrain=200, preq_samples=100000):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    n_samples = 0
    true_labels, pred_labels = [], []
    processing_times = []
    drift_idx_list = []
    pred_probabilities = []

    # pretraining phase
    if isinstance(classifier, AdaptiveRandomForestClassifier) or isinstance(classifier, KNNClassifier) or isinstance(classifier, KNNADWINClassifier):
        # for _ in range(9900):
        #     stream.next_sample()
        stream.next_sample(9900)
    
        for _ in range(n_pretrain):
            X_pretrain, y_pretrain = stream.next_sample()
            classifier.partial_fit(X_pretrain, y_pretrain, classes=stream.target_values)
        
        if isinstance(classifier, AdaptiveRandomForestClassifier):
            prev_drift_counts = [learner.nb_drifts_detected for learner in classifier.ensemble]
    else:
        for _ in range(n_pretrain):
            X_pretrain, _ = stream.next_sample()
            classifier.learn_one(dict(enumerate(*X_pretrain)))
        
        stream.next_sample(9900)
        

    # prequential loop
    while n_samples < preq_samples and stream.has_more_samples():
        X, y = stream.next_sample()
        try:
            if X is not None and y is not None:
                start = timer()
                n_samples += 1

                # with dynamic feature selection
                if feature_selector is not None:
                    feature_selector.weight_features(copy.copy(X), copy.copy(y))
                    X_select = feature_selector.select_features(copy.copy(X), rng=np.random.default_rng())
                    
                    # Test first
                    if isinstance(classifier, AdaptiveRandomForestClassifier) or isinstance(classifier, KNNClassifier) or isinstance(classifier, KNNADWINClassifier):
                        y_pred = classifier.predict(X_select)
                        y_prob = classifier.predict_proba(X_select)
                        if y_prob.shape[1] >= 2:
                            pred_probabilities.append(y_prob[0][1])  # Probability of positive class
                        else:
                            pred_probabilities.append(0.0)                    
                    # OC-SVM
                    else:
                        score = classifier.score_one(dict(enumerate(*X_select)))
                        pred_probabilities.append(score)
                        y_pred = classifier.classify(score)
                        
                    
                    # Train incrementally
                    if isinstance(classifier, AdaptiveRandomForestClassifier) or isinstance(classifier, KNNClassifier) or isinstance(classifier, KNNADWINClassifier):
                        classifier.partial_fit(copy.copy(X_select), [y[0]])
                    else:
                        if y_pred == 0: # train only on "benign" samples
                            classifier.learn_one(copy.copy(dict(enumerate(*X_select))))
                
                # no feature selection
                else:
                    # Test first
                    if isinstance(classifier, AdaptiveRandomForestClassifier) or isinstance(classifier, KNNClassifier) or isinstance(classifier, KNNADWINClassifier):
                        y_pred = classifier.predict(X)
                        y_prob = classifier.predict_proba(X)
                        # print(y_prob)
                        if y_prob.shape[1] >= 2:
                            pred_probabilities.append(y_prob[0][1])  # Probability of positive class
                        else:
                            pred_probabilities.append(0.0) 
                    else:
                        score = classifier.score_one(dict(enumerate(*X)))
                        pred_probabilities.append(score)
                        y_pred = classifier.classify(score)
                    
                    # Train incrementally
                    if isinstance(classifier, AdaptiveRandomForestClassifier) or isinstance(classifier, KNNClassifier) or isinstance(classifier, KNNADWINClassifier):
                        classifier.partial_fit(copy.copy(X), [y[0]])
                    else:
                        if y_pred == 0: # train only on "benign" samples
                            classifier.learn_one(copy.copy(dict(enumerate(*X))))
                
                # drift detection
                if isinstance(drift_detection, ADWIN):
                    if isinstance(classifier, AdaptiveRandomForestClassifier):
                        for i, estimator in enumerate(classifier.ensemble):
                            if estimator.nb_drifts_detected > prev_drift_counts[i]:
                                drift_idx_list.append(classifier.instances_seen + 9900 - 1)
                                prev_drift_counts[i] = estimator.nb_drifts_detected

                    elif isinstance(classifier, KNNADWINClassifier):
                        if classifier.adwin.detected_change():
                            drift_idx_list.append(n_samples + 9900 - 1)
                    else: # one class svm
                        drift_detection.add_element(np.float64(y_pred == y))
                        if drift_detection.detected_change():
                            drift_idx_list.append(n_samples + 9900 - 1)
                            drift_detection.reset()
                            # reset one class svm model
                            classifier.anomaly_detector = (
                                fx.RBFSampler(gamma=classifier.anomaly_detector[0].gamma) | OneClassSVM(classifier.anomaly_detector[1].nu) # change with the optimized params
                            )

                # evaluation
                true_labels.append(y[0])
                if isinstance(classifier, AdaptiveRandomForestClassifier) or isinstance(classifier, KNNClassifier) or isinstance(classifier, KNNADWINClassifier):
                    pred_labels.append(y_pred[0])
                else:
                    pred_labels.append(y_pred)
                
                end = timer()
                processing_times.append(end - start)

        except BaseException as e:
            print(e)
            break

    # evaluation metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    roc_auc = roc_auc_score(true_labels, pred_probabilities)
    roc_fpr, roc_tpr, _ = roc_curve(true_labels, pred_probabilities)
    avg_processing_time = sum(processing_times) / len(processing_times)

    drift_idx_list = sorted(set(drift_idx_list))

    if feature_selector is None:
        return accuracy, precision, recall, f1, roc_auc, roc_fpr, roc_tpr, avg_processing_time, drift_idx_list
    else:
        weights_history = feature_selector.weights_history
        selection_history = feature_selector.selected_features_history
        return accuracy, precision, recall, f1, roc_auc, roc_fpr, roc_tpr, avg_processing_time, selection_history, weights_history, drift_idx_list