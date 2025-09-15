import os
import random
import numpy as np
import logging
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms

def set_logger(log_path):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)    

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)
      
def save_plt(x,y,xlabel,ylabel,filename):
    plt.plot(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_data(data_path, dataset_name):
    """This function downloads the specified dataset with proper transforms"""
    dataset_name = dataset_name.upper()
    if hasattr(torchvision.datasets, dataset_name):
        if dataset_name == "CIFAR10":
            # 🔥 Add augmentation + correct normalization
            T = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.247, 0.243, 0.261)
                ),
            ])
        elif dataset_name == "MNIST":
            T = transforms.ToTensor()

        train_data = datasets.__dict__[dataset_name](
            root=data_path, train=True, download=True, transform=T
        )
        test_data = datasets.__dict__[dataset_name](
            root=data_path, train=False, download=True, transform=T
        )
    else:
        raise AttributeError(f"...dataset \"{dataset_name}\" not supported!")

    # unsqueeze channel dimension for grayscale datasets
    if train_data.data.ndim == 3:
        train_data.data.unsqueeze_(3)

    return train_data, test_data

# 🔥 Fixed normalization inside client data loader
class load_data(Dataset):
    def __init__(self, x, y):
        self.length = x.shape[0]
        self.x = x.permute(0,3,1,2).float() / 255.0  # scale [0,255] → [0,1]
        self.y = y
        self.image_transform = transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.247, 0.243, 0.261)
        )
        
    def __getitem__(self, index):
        image, label = self.x[index], self.y[index]
        image = self.image_transform(image)
        return image, label
        
    def __len__(self):
        return self.length
           
def tensor_to_numpy(data, device):
    if device.type == "cpu":
        return data.detach().numpy()
    else:
        return data.cpu().detach().numpy()

def numpy_to_tensor(data, device, dtype="float"):
    if dtype == "float":
        return torch.tensor(data, dtype=torch.float).to(device)
    elif dtype == "long":
        return torch.tensor(data, dtype=torch.long).to(device)

def evaluate_fn(dataloader, model, loss_fn, device):
    model.eval()
    running_loss, total, correct = 0, 0, 0
    with torch.no_grad():
        for batch, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = loss_fn(output, labels)
            running_loss += loss.item()
            total += labels.size(0)
            correct += (output.argmax(dim=1) == labels).sum().item()
    avg_loss = running_loss/(batch+1)
    acc = 100 * (correct/total)
    return avg_loss, acc