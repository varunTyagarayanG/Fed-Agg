import logging

# Place this at the very top of train.py
logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def train(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(dataloader, 1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Console + File logging for batch loss
        if batch_idx % 100 == 0:  # adjust frequency if needed
            msg = f"Epoch {epoch:03d} Batch {batch_idx:04d}/{len(dataloader)} - Loss: {loss.item():.4f}"
            print(msg)
            logging.info(msg)

    avg_loss = running_loss / len(dataloader)
    msg = f"Epoch {epoch:03d}: Train Loss = {avg_loss:.4f}"
    print(msg)
    logging.info(msg)
