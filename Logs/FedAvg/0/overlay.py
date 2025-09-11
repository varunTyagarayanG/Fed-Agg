import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Define folder structure
folders = {
    "10 Clients": "10 clients",
    "30 Clients": "30 clients",
    "50 Clients": "50 clients"
}

# Colors for the plots
colors = {
    "10 Clients": "blue",
    "30 Clients": "green",
    "50 Clients": "red"
}

# Extract accuracy from a single log file
def extract_accuracy(file_path):
    accuracies = []
    pattern = r'Accuracy:\s*([\d\.]+)%'
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                accuracies.append(float(match.group(1)))
    return accuracies

# Initialize plot
plt.figure(figsize=(12, 8))
data_found = False

for label, folder in folders.items():
    folder_path = os.path.join(os.getcwd(), folder)
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder}' does not exist.")
        continue

    # Collect all log files in the folder
    log_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".txt")]
    if not log_files:
        print(f"Error: No log files found in folder '{folder}'")
        continue

    all_accuracies = []
    for log_file in log_files:
        acc = extract_accuracy(log_file)
        if acc:
            all_accuracies.append(acc)
    
    if not all_accuracies:
        print(f"Warning: No accuracy data found in folder '{folder}'")
        continue

    # Make sure all logs have the same length
    min_len = min(len(acc) for acc in all_accuracies)
    all_accuracies = [acc[:min_len] for acc in all_accuracies]

    # Average across clients
    avg_accuracy = np.mean(all_accuracies, axis=0)

    rounds = list(range(1, len(avg_accuracy) + 1))
    plt.plot(rounds, avg_accuracy, label=label, color=colors[label])
    print(f"{label}: Averaged {len(avg_accuracy)} accuracy points from {len(log_files)} logs")
    data_found = True

if not data_found:
    print("No data available to plot. Please check your folders and log files.")
else:
    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Overlay of Accuracy Curves for Different Client Counts")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_file = "overlay_accuracy.png"
    plt.savefig(output_file)
    print(f"Overlay graph saved as {output_file}")
    plt.show()
