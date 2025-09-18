import torch
from copy import deepcopy

class Client:
    def __init__(self, client_id, local_data, device, num_epochs, lr, criterion):
        self.id = client_id
        self.data = local_data
        self.device = device
        self.num_epochs = num_epochs
        self.lr = lr
        self.criterion = criterion

        self.model = None
        self.gradient_x = None  # required for MimeLite
        self.delta = None       # client model delta

    def set_model(self, model):
        """Receive global model"""
        self.model = deepcopy(model).to(self.device)

    def client_update(self):
        """Perform local training and compute gradients (MimeLite needs gradients + delta)."""
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        for _ in range(self.num_epochs):
            for x, y in self.data:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()

                # Store gradients for MimeLite
                grads = [p.grad.detach().clone() for p in self.model.parameters()]
                self.gradient_x = grads

                optimizer.step()

        # Compute delta (local_model - global_model)
        self.delta = [p.data.detach().clone() for p in self.model.parameters()]

        return self.delta, self.gradient_x
