import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

def plot_and_save_confusion_matrix(json_path):
    # Load cls_labels and cls_preds from JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    cls_labels = data['cls_labels']
    cls_preds = data['cls_preds']

    # Ensure matrix is 5x5 even if only subset of labels is used
    all_classes = [0, 1, 2, 3]
    cm = confusion_matrix(cls_labels, cls_preds, labels=all_classes)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=all_classes,
                yticklabels=all_classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    # Save the confusion matrix image in the same directory
    save_dir = os.path.dirname(json_path)
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path)
    plt.close()

    print(f"Confusion matrix saved to: {save_path}")

def plot_confusion_matrix_spice(csv_path):
    # Load CSV
    df = pd.read_csv(csv_path)
    y_true = df["category"]
    y_pred = df["predicted_score"]

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", 
                xticklabels=[0, 1, 2, 3], 
                yticklabels=[0, 1, 2, 3])
    plt.xlabel("Predicted Score")
    plt.ylabel("True Category")
    plt.title("Confusion Matrix")

    # Save in same folder as CSV
    save_path = os.path.join(os.path.dirname(csv_path), "confusion_matrix.png")
    plt.savefig(save_path, dpi=300)

# Example usage
if __name__ == "__main__":
    # plot_and_save_confusion_matrix('exp_results/2025-07-10_01-25-04_MTL_E10_cls=5_e=50_bs=1_ctcW=0.1/test_metric_results.json')
    # plot_confusion_matrix_spice("spice_things/spice_results_youtube_noiseReduce.csv")
    # plot_confusion_matrix_spice("s3prl_things/test_predictions.csv")
    path = r"wav2vec_things\models\pseudo1\wav2vec_test_predictions.csv"
    plot_confusion_matrix_spice(path)

