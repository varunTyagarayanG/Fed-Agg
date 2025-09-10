import logging
import torch.nn as nn
import torch
import logging
criterion = nn.CrossEntropyLoss()

def test(model, dataloader, device):
    model.eval()
    correct, total, test_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = test_loss / len(dataloader)

    msg = f"Test Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.2f}%"
    print(msg)
    logging.info(msg)
