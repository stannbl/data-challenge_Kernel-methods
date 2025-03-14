import numpy as np

def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))


def kernel_logistic_regression(K_train, Y_train, K_test, lr=1, max_iter=1000, lambda_reg=0.01):
    m = K_train.shape[0]
    alpha = np.zeros(m)

    for i in range(max_iter):
        y_pred = 1 / (1 + np.exp(-K_train @ alpha))  # Sigmoid function
        gradient = K_train.T @ (y_pred - Y_train) / m + lambda_reg * alpha  # Regularized gradient
        alpha -= lr * gradient  # Update alpha

        if i % 100 == 0:
            lr *= 0.95  # Reduce learning rate over time (adaptive updates)

    return 1 / (1 + np.exp(-K_test @ alpha))  # Predict probabilities


