"""
DEPRECATED: Optional threshold helper not required for the minimal experiment.
Canonical entry: `src/main.py` → `experiments.tgn_svdd_experiment`.
Kept for reference only.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm




def gmm_classifier(train_data, test_data, percentage=5, n_components=4):
    
    # Check if the input data is in list format and convert it to NumPy arrays
    if isinstance(train_data, list):
        train_data = np.array(train_data)
    if isinstance(test_data, list):
        test_data = np.array(test_data)
    
    # Fit the GMM to the training data
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(train_data.reshape(-1, 1))

    # Calculate the log-likelihood of the training data points
    log_likelihoods = gmm.score_samples(train_data.reshape(-1, 1))

    # Calculate the threshold based on the desired percentage
    threshold = np.percentile(log_likelihoods, percentage)

    # Calculate the log-likelihood of the test data points
    test_log_likelihoods = gmm.score_samples(test_data.reshape(-1, 1))

    # Classify the test data points based on the threshold
    predicted_labels = (test_log_likelihoods < threshold).astype(int)
    
    return predicted_labels#, gmm.predict_proba(test_data.reshape(-1, 1)).sum(axis=1)




