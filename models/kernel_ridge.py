import numpy as np


def kernel_ridge_regression(K_train, Y_train, K_test, alpha=1e-3):
    """Solve (K + alpha*I) alpha = Y for Kernel Ridge Regression"""
    n = K_train.shape[0]
    I = np.eye(n)
    alpha_vec = np.linalg.solve(K_train + alpha * I, Y_train)

    # Predict on test data
    Y_pred = np.dot(K_test, alpha_vec)
    return (Y_pred > 0.5).astype(int)  # Convert to 0/1 predictions
