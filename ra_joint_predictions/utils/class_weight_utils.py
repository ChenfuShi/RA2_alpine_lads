import logging

import numpy as np

def calc_default_class_weights(outcomes, no_classes):
    D = outcomes.shape[1]

    outcomes_class_weights = []

    for d in range(D):
        class_weights = {}
        
        classes, counts = np.unique(outcomes.iloc[:, d].to_numpy(), return_counts = True)

        weights = (1 / counts) * (np.sum(counts)) / 2.0

        for idx, c in enumerate(classes.astype(np.int64)):
            class_weights[c] = weights[idx]
           
        for class_val in np.arange(no_classes):
            if class_val not in class_weights.keys():                
                class_weights[class_val] = 1

        outcomes_class_weights.append(class_weights)

    return outcomes_class_weights

def calc_adapted_class_weights(outcomes, no_classes):
    outcomes = outcomes.to_numpy()
    
    D = outcomes.shape[1]

    outcomes_class_weights = []

    for d in range(D):
        class_weights = {}

        non0_idx = np.where(outcomes[:, d] != 0)[0]

        selection_probabilities = np.zeros(no_classes)
        # 0s have a 0.5 probability of being selected
        selection_probabilities[0] = 0.5

        classes, counts = np.unique(outcomes[non0_idx, d], return_counts = True)
        non_non0_probs = 0.5 * (counts / np.sum(counts))
        min_non0_prob = np.min(non_non0_probs)

        selection_probabilities[classes.astype(np.int32)] = non_non0_probs

        # Set prob for not found class to min of class non 0
        for missing_class in np.arange(1, no_classes):
            if missing_class not in classes:
                selection_probabilities[missing_class] = min_non0_prob

        weights = (1 / selection_probabilities) / 2

        min_non0_weight = np.min(weights[1:])

        non_outlier_idx = weights < 15 * min_non0_weight
        outlier_idxs = np.logical_not(non_outlier_idx)
        
        if np.count_nonzero(outlier_idxs) > 0:
            mean_outlier_val = np.mean(weights[non_outlier_idx])
            std_outlier_val = np.std(weights[non_outlier_idx])

            outlier_val = mean_outlier_val + 3 * std_outlier_val

            logging.info("Outcome %d, found outliers weights %s", d, weights[outlier_idxs])

            weights[np.logical_not(non_outlier_idx)] = outlier_val
            logging.info("Outcome %d, Updated outlier weights: %s", d, weights[outlier_idxs])

        for idx, c in enumerate(weights):
            class_weights[idx] = c

        outcomes_class_weights.append(class_weights)

    return outcomes_class_weights