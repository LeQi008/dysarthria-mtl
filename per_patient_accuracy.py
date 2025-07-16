
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

def extract_patient_code(name):
    # Match Torgo types that have "Session" in their name
    if "Session" in name:
        # Case 1: Name starts with patient ID like 'M01_Session...'
        return name.split("_")[0]
    else:
        # Match "audio 1_speaker 3" → extract 1 and 3
        match = re.search(r'audio\s*(\d+)_speaker\s*(\d+)', name, re.IGNORECASE)
        if match:
            audio_id = match.group(1)
            speaker_id = match.group(2)
            return f"A{audio_id}S{speaker_id}"
        return "Unknown"

def plot_patient_accuracies(csv_path, output_path="patient_accuracy_barplot.png"):
    df = pd.read_csv(csv_path)
    df["patient_id"] = df["name"].apply(extract_patient_code)

    # Compute per-patient accuracy
    summary = (
        df.groupby("patient_id")
        .apply(lambda g: pd.Series({
            "accuracy": (g["category"] == g["predicted_score"]).mean() * 100,
            "total": len(g),
            "correct": (g["category"] == g["predicted_score"]).sum(),
            "label": g["category"].iloc[0]
        }))
        .reset_index()
    )

    # Sort by true label (ascending), then accuracy (descending)
    summary = summary.sort_values(by=["label", "accuracy"], ascending=[False, False])

    # Normalize accuracy to 0–1 for color saturation
    norm_accuracies = summary["accuracy"] / 100
    colors = sns.color_palette("Blues", len(summary))
    cmap = sns.light_palette("blue", n_colors=100)

    # Get color per bar based on accuracy
    bar_colors = [cmap[int(acc)] for acc in norm_accuracies * 99]

    # Plot bar chart
    plt.figure(figsize=(10, max(4, 0.4 * len(summary))))

    # Create display labels: A1S3 (label = 2)
    summary["label_str"] = summary.apply(
        lambda row: f"{row['patient_id']} (label = {row['label']})", axis=1
    )
    # Plot with custom y-axis labels
    bars = plt.barh(summary["label_str"], summary["accuracy"], color=bar_colors)


    # Annotate bars
    for bar, acc in zip(bars, summary["accuracy"]):
        xpos = bar.get_width()
        label = f"{acc:.1f}%"
        if xpos > 10:
            plt.text(xpos - 5, bar.get_y() + bar.get_height() / 2, label,
                     va='center', ha='right', color='white', fontsize=9)
        else:
            plt.text(xpos + 1, bar.get_y() + bar.get_height() / 2, label,
                     va='center', ha='left', color='black', fontsize=9)

    plt.xlabel("Accuracy (%)")
    plt.ylabel("Patient ID")
    plt.title("Per-Patient Prediction Accuracy (Sorted by Severity Label)")
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"[✓] Saved plot to: {output_path}")

import pandas as pd

def print_patient_accuracies(csv_path):
    df = pd.read_csv(csv_path)

    # Add patient ID
    df["patient_id"] = df["name"].apply(extract_patient_code)

    # Group and compute per-patient accuracy
    patient_stats = []
    total_correct = 0
    total_count = 0

    for patient_id, group in df.groupby("patient_id"):
        total = len(group)
        correct = (group["category"] == group["predicted_score"]).sum()
        accuracy = correct / total * 100
        true_label = group["category"].iloc[0]
        patient_stats.append((true_label, patient_id, correct, total, accuracy))

        total_correct += correct
        total_count += total

    # Sort by true label (ascending), then accuracy descending
    patient_stats.sort(key=lambda x: (x[0], -x[4]))

    # Print per-patient results
    for true_label, patient_id, correct, total, accuracy in patient_stats:
        print(f"{patient_id} (Label {true_label}): {correct}/{total} correct → {accuracy:.2f}%")

    # Print total accuracy
    total_accuracy = total_correct / total_count * 100
    print(f"\nTotal: {total_correct}/{total_count} correct → {total_accuracy:.2f}%")

    
# plot_patient_accuracies("spice_things/spice_results_youtube.csv")
print_patient_accuracies("spice_things/spice_results_torgo1000_over10Char.csv")
