import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def load_data(file_path):
    """Load predictions or truth from file"""
    if not os.path.exists(file_path):
        print(f"Error: The file at {file_path} does not exist.")
        sys.exit(0)
    try:
        with open(file_path, 'r') as f:
            result = []
            for line in f.readlines():
                r = line.strip()
                if r == 'Jedi' or r == 'Sith':
                    result.append(r)
                else:
                    print(f"Error: The file content **{r}** is invalid.\n")
                    sys.exit(0)
            if len(result) != 100:
                print(f"Erro: The file len **{len(result)}** is invalid\n")
                sys.exit()
            return result
    except PermissionError:
        print(f"Error: Permission denied when accessing {file_path}.")
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(0)


def confusion_matrix(y_true, y_pred):
    """Calculate confusion matrix manually"""
    n = 2
    matrix = np.zeros((n, n), dtype=int)
    label_to_idx = {'Jedi': 0, 'Sith': 1}
    for true, pred in zip(y_true, y_pred):
        true_idx = label_to_idx[true]
        pred_idx = label_to_idx[pred]
        matrix[true_idx][pred_idx] += 1
    return matrix


def calculate_metrics(confusion_mat):
    """Calculate precision, recall, f1-score, and accuracy"""
    labels = ['Jedi', 'Sith']
    metrics = {}
    total_correct = 0
    total_samples = 0
    for i, label in enumerate(labels):
        tp = confusion_mat[i][i]
        fp = np.sum(confusion_mat[:, i]) - tp
        fn = np.sum(confusion_mat[i, :]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        p = 2 * (precision * recall) / (precision + recall)
        f1 = p if (precision + recall) > 0 else 0
        support = tp + fn
        metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': support
        }
        total_correct += tp
        total_samples += support
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return metrics, accuracy


def display_confusion_matrix(confusion_mat):
    """Display confusion matrix as heatmap."""
    labels = ['Jedi', 'Sith']
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_mat, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    max_val = confusion_mat.max()

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = confusion_mat[i, j]
            text_color = "white" if value > max_val / 2 else "black"
            plt.text(
                j, i, str(value),
                ha="center",
                va="center",
                color=text_color
            )

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    # plt.show()
    plt.close()


def main():
    if len(sys.argv) != 3:
        print("Usage: python3.10 Confusion_Matrix.py ../dataset/predictions.txt ../dataset/truth.txt")
        sys.exit(1)

    pred_file = sys.argv[1]
    truth_file = sys.argv[2]
     

    predictions = load_data(pred_file)
    truth = load_data(truth_file)

   
    conf_matrix = confusion_matrix(truth, predictions)
    metrics, accuracy = calculate_metrics(conf_matrix)

    print(f"{'':10} precision    recall  f1-score     total")
    labels = ['Jedi', 'Sith']
    for label in labels:
        m = metrics[label]
        print(
            f"{label:10}"
            f"     {m['precision']:.2f}"
            f"      {m['recall']:.2f}"
            f"      {m['f1-score']:.2f}"
            f"        {m['support']}"
        )

    total_samples = sum(m['support'] for m in metrics.values())

    print(
        f"\n{'accuracy':10}"
        f"                       {accuracy:.2f}"
        f"       {total_samples}"
    )
    display_confusion_matrix(conf_matrix)


if __name__ == "__main__":
    main()
