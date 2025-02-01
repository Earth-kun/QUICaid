import numpy as np
import copy

def run_prequential(setup_name, classifier, stream, drift_detector, feature_selector, n_pretrain=200):
    """
    Parameters
    ----------
    setup (str): 
    classifier
    etc...
    """
    stream.restart()
    n_samples, correct_predictions = 0, 0
    true_labels, pred_labels = [], []

    print(f"Evaluating {setup_name} configuration.")

    # pretrain samples
    X_pretrain, y_pretrain = stream.next_sample(n_pretrain)
    classifier.partial_fit(X_pretrain, y_pretrain, classes=stream.target_values)
    
    print(f"Model pretrained on {n_pretrain} samples.")

    while n_samples < 100000 and stream.has_more_samples():
        X, y = stream.next_sample()
        n_samples += 1

        if feature_selector is not None:
            # with dynamic feature selection
            feature_selector.weight_features(copy.copy(X), copy.copy(y))
            X_select = feature_selector.select_features(copy.copy(X), rng=np.random.default_rng())
            y_pred = classifier.predict(X_select)
            
            # Train incrementally
            classifier.partial_fit(copy.copy(X_select), [y[0]])

        else:
            # no feature selection
            y_pred = classifier.predict(X)
            
            # Train incrementally
            classifier.partial_fit(copy.copy(X), [y[0]])
        
        if y_pred == y:
            correct_predictions += 1
        
        true_labels.append(y[0])
        pred_labels.append(y_pred[0])

        # check for drift
        if drift_detector is not None:
            drift_detector.add_element(np.float64(y_pred == y))
            if drift_detector.detected_change():
                print(f"drift detected at {n_samples}")


    # Calculate accuracy
    accuracy = correct_predictions / n_samples
    print(accuracy)