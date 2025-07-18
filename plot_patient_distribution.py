import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

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

def plot_patient_prediction_matrix(csv_path):
    # Load CSV
    df = pd.read_csv(csv_path)

    # ===== Youtube Audio & Torgo =====
    # Extract compact patient ID
    df["patient_id"] = df["name"].apply(extract_patient_code)
    # Create display label: "A1S3 (label 2)"
    df["patient_label"] = df.apply(lambda row: f"{row['patient_id']} (label {row['category']})", axis=1)


    # Create a new DataFrame with patient and category for sorting
    patient_order = (
        df[["patient_label", "category"]]
        .drop_duplicates()
        .sort_values(by="category")
        .set_index("patient_label")
    )

    # Build pivot table: rows = patients, columns = predicted_score
    count_matrix = (
        df.groupby(["patient_label", "predicted_score"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(5), fill_value=0)
    )

    # Reorder rows by category (ascending)
    count_matrix = count_matrix.loc[patient_order.index]

    # Plot heatmap
    plt.figure(figsize=(10, max(4, 0.4 * len(count_matrix))))
    sns.heatmap(
        count_matrix,
        annot=True,
        fmt='d',
        cmap="Blues",             # Blue palette, luminance-based
        cbar_kws={"label": "Count"},
        linewidths=0.5,
        linecolor='white'
    )

    plt.title("Predicted Severity Distribution per Patient")
    plt.xlabel("Predicted Score")
    plt.ylabel("Patient (True Label)")
    plt.tight_layout()

    # Save in same folder as CSV
    save_path = os.path.join(os.path.dirname(csv_path), "patient_prediction_matrix.png")
    plt.savefig(save_path, dpi=300)

    print(f"[✓] Matrix saved to: {save_path}")

if __name__ == "__main__":
    # plot_patient_prediction_matrix("spice_things/spice_results_youtube_noiseReduce.csv")
    plot_patient_prediction_matrix("s3prl_things/test_predictions.csv")
