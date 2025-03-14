import numpy as np
import cvxopt


class SVM:
    def __init__(self, C=1.0):
        """
        Initialize the SVM model with a regularization parameter C.
        """
        self.C = C
        self.alpha = None  # Lagrange multipliers
        self.sv = None  # Support vectors
        self.sv_y = None  # Support vector labels
        self.sv_alpha = None  # Nonzero alphas
        self.b = 0  # Bias term

    def fit(self, K, y, epsilon=1e-5):
        """
        Train the SVM using a precomputed kernel matrix with regularization.

        Parameters:
        - K: Precomputed kernel matrix of shape (n_samples, n_samples)
        - y: Labels of shape (n_samples,), expected in {-1, 1}
        - epsilon: Small regularization term to stabilize the kernel matrix.
        """
        n_samples = K.shape[0]

        # Convert labels to a column vector
        y = y.astype(np.double).reshape(-1, 1)

        # Regularize kernel matrix (add small epsilon to diagonal)
        K += epsilon * np.eye(n_samples)

        # Setup quadratic programming problem
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones((n_samples, 1)))
        G = cvxopt.matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
        A = cvxopt.matrix(y.T, tc='d')
        b = cvxopt.matrix(0.0)

        # Solve the quadratic problem
        cvxopt.solvers.options['show_progress'] = False

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        alpha = np.ravel(solution['x'])

        # Select support vectors
        sv = alpha > 1e-5
        self.sv = np.where(sv)[0]
        self.sv_alpha = alpha[sv]
        self.sv_y = y[sv].reshape(-1)

        # Bias term
        self.b = np.mean(self.sv_y - np.sum(self.sv_alpha * self.sv_y * K[self.sv][:, self.sv], axis=1))

    def predict(self, K_test):
        """
        Predict class labels for given test samples using the precomputed kernel.

        Parameters:
        - K_test: Kernel matrix between test samples and support vectors.

        Returns:
        - y_pred: Predicted labels in {-1, 1}
        """
        decision_values = np.sum(self.sv_alpha * self.sv_y * K_test[:, self.sv], axis=1) + self.b
        return np.sign(decision_values)
