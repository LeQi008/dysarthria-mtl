import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("training_stats.csv")

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
plt.savefig("training_curves.png")
plt.close()

print("✅ Saved plot as 'training_curves.png'")
