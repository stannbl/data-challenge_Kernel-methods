import pandas as pd
import numpy as np
from scripts.preprocessing_advanced import transform_sequences
from models.kernel_methods import mismatch_kernel
from models.kernel_svm import SVM
import os

# Define datasets
datasets = [0, 1, 2]

# Load training sequences & labels
X_train = {k: pd.read_csv(f"data/Xtr{k}.csv", header=0)["seq"] for k in datasets}
Y_train = {k: pd.read_csv(f"data/Ytr{k}.csv", header=0)["Bound"].astype(float) for k in datasets}
X_test = {k: pd.read_csv(f"data/Xte{k}.csv", header=0).iloc[:, 1] for k in datasets}

# Transform sequences using advanced preprocessing
X_train_transformed = {dataset_id: transform_sequences(X_train[dataset_id], k=6) for dataset_id in datasets}
X_test_transformed = {dataset_id: transform_sequences(X_test[dataset_id], k=6) for dataset_id in datasets}


# Function to shuffle and split data into training and validation sets
def train_val_split(X, Y, split_ratio=0.8):
    n_samples = min(len(X), len(Y))
    perm = np.random.permutation(n_samples)
    X, Y = np.array(X[:n_samples])[perm], np.array(Y[:n_samples])[perm]  # Ensure shuffled order
    split_index = int(len(X) * split_ratio)
    return X[:split_index], X[split_index:], Y[:split_index], Y[split_index:]


X_train_split, X_val_split, Y_train_split, Y_val_split = {}, {}, {}, {}

for k in datasets:
    X_train_split[k], X_val_split[k], Y_train_split[k], Y_val_split[k] = train_val_split(
        X_train_transformed[k], Y_train[k]
    )

# Convert Y labels to {-1,1} for SVM
for k in datasets:
    Y_train_split[k] = (2 * np.array(Y_train_split[k]) - 1).reshape(-1)
    Y_val_split[k] = (2 * np.array(Y_val_split[k]) - 1).reshape(-1)

# Hyperparameters
k_values = [6]
m_values = [3]
C_values = [20]

# k=6, m=4, C=50 → 0.65583
# k=6, m=3, C=20 → 0.66333

best_params = None
best_acc = 0

print("🔍 Starting hyperparameter tuning...")

for k in k_values:
    for m in m_values:

        # Compute kernel matrices for each dataset
        K_train_dict = {}
        K_val_dict = {}

        for d in datasets:
            K_train_dict[d] = mismatch_kernel(X_train_split[d], X_train_split[d], k=k, m=m)
            K_val_dict[d] = mismatch_kernel(X_val_split[d], X_train_split[d], k=k, m=m)

        # Train SVM and evaluate
        for C in C_values:
            correct_predictions = 0
            total_samples = 0

            for d in datasets:

                # Train SVM model
                svm = SVM(C=C)
                svm.fit(K_train_dict[d], Y_train_split[d])
                Y_pred_val = svm.predict(K_val_dict[d])

                correct_predictions += np.sum(Y_pred_val == Y_val_split[d])
                total_samples += len(Y_val_split[d])

            acc = correct_predictions / total_samples
            print(f"✅ Accuracy: k={k}, m={m}, C={C} → {acc:.5f}")

            if acc > best_acc:
                best_acc = acc
                best_params = (k, m, C)

# Print best parameters
print(f"🏆 Best params: k={best_params[0]}, m={best_params[1]}, C={best_params[2]}, Accuracy={best_acc:.5f}")

# Retrain using the full dataset with best parameters
print("🚀 Training final model on full dataset...")
Y_pred_list = []

for k in datasets:
    print(f"⚡ Computing mismatch kernel for full dataset {k}, k={best_params[0]}, m={best_params[1]}...")

    K_train = mismatch_kernel(X_train_transformed[k], X_train_transformed[k], k=best_params[0], m=best_params[1])
    K_test = mismatch_kernel(X_test_transformed[k], X_train_transformed[k], k=best_params[0], m=best_params[1])

    print(f"⚡ Training final SVM model for dataset {k}...")
    svm = SVM(C=best_params[2])
    svm.fit(K_train, (2 * Y_train[k].to_numpy() - 1))  # Convert labels to {-1, 1}
    Y_pred_k = svm.predict(K_test)

    # Convert predictions back to {0, 1}
    Y_pred_k = (Y_pred_k + 1) // 2
    Y_pred_list.append(Y_pred_k)

# Concatenate all predictions
Y_pred = np.concatenate(Y_pred_list, axis=0)
submission_ids = np.concatenate([np.arange(1000 * k, 1000 * (k + 1)) for k in datasets])

# Create submission file
submission = pd.DataFrame({"Id": submission_ids, "Bound": Y_pred.flatten().astype(int)})
submission.to_csv("Yte.csv", index=False)

print("🎉 Final submission saved as Yte.csv")
