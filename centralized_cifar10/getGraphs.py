import re
import matplotlib.pyplot as plt
import os

# Path to your log file
log_dir = os.path.join(os.path.dirname(__file__), "logs")
log_file = os.path.join(log_dir, "training.log")

# Regex patterns
batch_pattern = re.compile(r"Epoch (\d+) Batch \d+/\d+ - Loss: ([\d\.]+)")
train_pattern = re.compile(r"Epoch (\d+): Train Loss = ([\d\.]+)")
test_pattern = re.compile(r"Test Loss: ([\d\.]+) \| Test Accuracy: ([\d\.]+)%")

# Storage
batch_losses = {}
epoch_losses = []
test_losses = []
test_accuracies = []
epochs = []
test_epochs = []

# Track the current epoch for when test logs appear
current_epoch = None

# Read and parse log file
with open(log_file, "r") as f:
    for line in f:
        if m := batch_pattern.search(line):
            current_epoch = int(m.group(1))
            loss = float(m.group(2))
            batch_losses.setdefault(current_epoch, []).append(loss)

        if m := train_pattern.search(line):
            current_epoch = int(m.group(1))
            loss = float(m.group(2))
            epochs.append(current_epoch)
            epoch_losses.append(loss)

        if m := test_pattern.search(line):
            loss, acc = float(m.group(1)), float(m.group(2))
            if current_epoch is not None:
                test_epochs.append(current_epoch)
            else:
                test_epochs.append(len(test_epochs) * 10)  # fallback
            test_losses.append(loss)
            test_accuracies.append(acc)

# --- Create output folder for graphs ---
out_dir = os.path.join(log_dir, "graphs")
os.makedirs(out_dir, exist_ok=True)

# 1. Train Loss per Epoch
if epoch_losses:
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, epoch_losses, "o-", color="red", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Train Loss")
    plt.title("Train Loss per Epoch")
    plt.ylim(0, max(epoch_losses) + 0.5)
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "train_loss.png"), dpi=200)
    plt.close()

# 2. Test Loss
if test_losses:
    plt.figure(figsize=(10, 6))
    plt.plot(test_epochs, test_losses, "s--", color="blue", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Test Loss")
    plt.title("Test Loss")
    plt.ylim(0, max(test_losses) + 0.5)
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "test_loss.png"), dpi=200)
    plt.close()

# 3. Test Accuracy
if test_accuracies:
    plt.figure(figsize=(10, 6))
    plt.plot(test_epochs, test_accuracies, "d--", color="green", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy")
    plt.ylim(min(test_accuracies) - 2, max(test_accuracies) + 2)
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "test_accuracy.png"), dpi=200)
    plt.close()

# 4. Batch Losses (first few epochs only)
if batch_losses:
    plt.figure(figsize=(12, 7))
    for epoch, losses in batch_losses.items():
        if epoch <= 3:  # only first few for readability
            plt.plot(range(len(losses)), losses, alpha=0.6, label=f"Epoch {epoch}")
    plt.xlabel("Batch Index")
    plt.ylabel("Batch Loss")
    plt.title("Batch Loss Curves (First Few Epochs)")
    plt.ylim(0, max(max(v) for v in batch_losses.values() if v) + 0.5)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "batch_losses.png"), dpi=200)
    plt.close()

print(f"✅ Individual graphs saved in {out_dir}")
