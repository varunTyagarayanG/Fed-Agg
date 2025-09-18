import torch
from copy import deepcopy


class Client:
    def __init__(self, client_id, local_data, device, num_epochs, criterion, lr):
        self.id = client_id
        self.data = local_data              # expected to be a DataLoader
        self.device = device
        self.num_epochs = num_epochs
        self.lr = lr
        self.criterion = criterion
        self.x = None
        self.y = None
        self.delta_y = None
        self.state = None

    def client_update(self):
        # initialize local model from global x
        self.y = deepcopy(self.x).to(self.device)
        self.y.train()

        optimizer = torch.optim.SGD(self.y.parameters(), lr=self.lr)

        for epoch in range(self.num_epochs):
            for inputs, labels in self.data:
                inputs, labels = inputs.float().to(self.device), labels.long().to(self.device)

                optimizer.zero_grad()
                output = self.y(inputs)
                loss = self.criterion(output, labels)
                loss.backward()
                optimizer.step()

        # compute delta_y (model update)
        with torch.no_grad():
            delta_y = [p_y.detach().cpu() - p_x.detach().cpu()
                       for p_y, p_x in zip(self.y.parameters(), self.x.parameters())]
        self.delta_y = delta_y
