import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, accuracy_score
import matplotlib.pyplot as plt
import sys

def load_and_prepare_data(train_file, valid_file, test_file):
    """Load and prepare training, validation, and test data"""

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Detect target column
    target_cols = ['knight', 'Target', 'label', 'class']
    target_col = next((col for col in target_cols if col in train_df.columns), train_df.columns[-1])

    # Convert labels
    if train_df[target_col].dtype == 'object':
        train_df[target_col] = train_df[target_col].map({'Jedi': 1, 'Sith': 0})

    if target_col in test_df.columns and test_df[target_col].dtype == 'object':
        test_df[target_col] = test_df[target_col].map({'Jedi': 1, 'Sith': 0})

    # Split features/labels
    X_train = train_df.drop(target_col, axis=1)
    y_train = train_df[target_col]

    if target_col in test_df.columns:
        X_test = test_df.drop(target_col, axis=1)
        y_test = test_df[target_col]
    else:
        X_test = test_df
        y_test = None

    # Load validation
    X_val, y_val = None, None
    try:
        val_df = pd.read_csv(valid_file)

        if target_col in val_df.columns and val_df[target_col].dtype == 'object':
            val_df[target_col] = val_df[target_col].map({'Jedi': 1, 'Sith': 0})

        X_val = val_df.drop(target_col, axis=1)
        y_val = val_df[target_col]
    except Exception as e:
        print("⚠ Validation file issue:", e)

    # Keep only numeric
    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])

    if X_val is not None:
        X_val = X_val.select_dtypes(include=[np.number])

    # Align columns
    common_cols = X_train.columns.intersection(X_test.columns)
    if X_val is not None:
        common_cols = common_cols.intersection(X_val.columns)

    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    if X_val is not None:
        X_val = X_val[common_cols]

    return X_train, y_train, X_test, y_test, X_val, y_val


def find_optimal_k(X_train, y_train, X_val, y_val):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    best_k, best_f1 = 1, 0

    k_values = list(range(1, 31))
    accuracies = []
    f1_scores = []

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_scaled, y_train)

        preds = model.predict(X_val_scaled)

        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average='weighted')

        accuracies.append(acc)
        f1_scores.append(f1)

        if f1 > best_f1:
            best_f1 = f1
            best_k = k

        print(f"k={k} -> Accuracy={acc:.4f}, F1={f1:.4f}")

    print(f"\nBest k = {best_k} (F1={best_f1:.4f})")

    return best_k, k_values, accuracies

def plot_k_accuracy(k_values, accuracies):
    plt.figure(figsize=(8, 5))

    plt.plot(k_values, accuracies, marker='o')
    
    plt.xlabel('k values')
    plt.ylabel('Accuracy')
    plt.title('KNN Accuracy vs K')
    
    plt.grid(True)

    # Highlight best k
    best_idx = accuracies.index(max(accuracies))
    best_k = k_values[best_idx]
    
    plt.axvline(x=best_k, linestyle='--')
    plt.scatter(best_k, max(accuracies))

    plt.text(best_k, max(accuracies),
             f'  best k={best_k}',
             verticalalignment='bottom')

    plt.tight_layout()
    plt.savefig('knn_accuracy_plot.png')
    plt.close()

def main():
    if len(sys.argv) != 4:
        print("Usage: python3.10 KNN.py ../dataset/Training_knight.csv ../dataset/Validation_knight.csv ../dataset/Test_knight.csv")
        sys.exit(1)

    train_file = sys.argv[1]
    valid_file = sys.argv[2]
    test_file  = sys.argv[3]

    X_train, y_train, X_test, y_test, X_val, y_val = load_and_prepare_data(
        train_file, valid_file, test_file
    )

    # Choose k
    if X_val is not None:
        k, k_values, accuracies = find_optimal_k(X_train, y_train, X_val, y_val)

        # Plot results
        plot_k_accuracy(k_values, accuracies)
    else:
        k = 5

    # Train final model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # model = KNeighborsClassifier(n_neighbors=k)
    model = KNeighborsClassifier(n_neighbors=k, weights='distance')
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)

    # Save predictions
    label_map = {0: 'Sith', 1: 'Jedi'}

    with open("KNN.txt", "w") as f:
        for p in preds:
            f.write(label_map.get(p, str(p)) + "\n")

    print("✓ Predictions saved to KNN.txt")


if __name__ == "__main__":
    main()