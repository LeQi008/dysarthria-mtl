import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the CSV
csv_path = r'C:\Users\YIDAN\Desktop\projects\dysarthria-mtl-steal\s3prl_things\training_log.csv'

def generate_curves(csv_path):
    df = pd.read_csv(csv_path)

    epochs = df["epoch"]

    # Create a 1x2 plot: Loss and Accuracy
    plt.figure(figsize=(12, 5))

    # ---- Plot Loss ----
    plt.subplot(1, 2, 1)
    plt.plot(epochs, df["train_loss"], label="Train Loss", marker='o')
    plt.plot(epochs, df["val_loss"], label="Validation Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.grid(True)

    # ---- Plot Accuracy ----
    plt.subplot(1, 2, 2)
    plt.plot(epochs, df["train_acc"], label="Train Accuracy", marker='o')
    plt.plot(epochs, df["val_acc"], label="Validation Accuracy", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.grid(True)

    # Save the figure
    plt.tight_layout()
    # Save in same folder as CSV
    save_path = os.path.join(os.path.dirname(csv_path), "training_curves.png")
    plt.savefig(save_path, dpi=300)

    print("✅ Saved plot as 'training_curves.png'")

if __name__ == "__main__":
    generate_curves(csv_path)
