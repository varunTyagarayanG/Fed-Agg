import torch
import logging
from copy import deepcopy


class Client:
    def __init__(self, client_id, local_data, device, num_epochs, criterion, lr):
        self.client_id = client_id
        self.local_data = local_data
        self.device = device
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.lr = lr
        
        # Model states
        self.x = None  # Global model (received from server)
        self.y = None  # Local model after training
        self.gradient_x = None  # Gradients computed at global model

    def client_update(self):
        """Performs local training and computes gradients for MimeLite"""
        if self.x is None:
            raise ValueError(f"Global model not set for client {self.client_id}")
        
        # Initialize local model y as copy of global model x
        self.y = deepcopy(self.x).to(self.device)
        
        # Compute gradient at global model x (for MimeLite)
        self.compute_gradient_x()
        
        # Perform local training on y
        optimizer = torch.optim.SGD(self.y.parameters(), lr=self.lr)
        
        self.y.train()
        total_loss = 0.0
        total_batches = 0
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(self.local_data):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.y(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            if num_batches > 0:
                avg_epoch_loss = epoch_loss / num_batches
                total_loss += avg_epoch_loss
                total_batches += 1
                logging.debug(f"Client {self.client_id}, Epoch {epoch+1}, Loss: {avg_epoch_loss:.6f}")
        
        if total_batches > 0:
            avg_total_loss = total_loss / total_batches
            logging.info(f"Client {self.client_id} completed training, Avg Loss: {avg_total_loss:.6f}")

    def compute_gradient_x(self):
        """Computes gradient at global model x using local data"""
        self.x.train()
        self.x.zero_grad()
        
        total_samples = 0
        accumulated_grads = [torch.zeros_like(p) for p in self.x.parameters()]
        
        # Compute gradient over all local data
        for batch_idx, (data, target) in enumerate(self.local_data):
            data, target = data.to(self.device), target.to(self.device)
            batch_size = data.size(0)
            
            # Forward pass
            output = self.x(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            self.x.zero_grad()
            loss.backward()
            
            # Accumulate gradients
            for i, param in enumerate(self.x.parameters()):
                if param.grad is not None:
                    accumulated_grads[i] += param.grad.data * batch_size
            
            total_samples += batch_size
        
        # Average gradients
        if total_samples > 0:
            self.gradient_x = [grad / total_samples for grad in accumulated_grads]
        else:
            self.gradient_x = [torch.zeros_like(p) for p in self.x.parameters()]
        
        logging.debug(f"Client {self.client_id} computed gradient_x over {total_samples} samples")