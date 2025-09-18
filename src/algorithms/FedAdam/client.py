import torch
from copy import deepcopy

class Client:
    """
    Local client in federated learning.
    """
    def __init__(self, client_id, local_data, device, num_epochs, criterion, lr):
        self.id = client_id
        self.data = local_data
        self.device = device
        self.num_epochs = num_epochs
        self.lr = lr
        self.criterion = criterion
        self.x = None  # Global model (received from server)
        self.y = None  # Local model (copy for training)
        self.delta_y = None  # Model updates to send back

    def client_update(self):
        """Perform local training and compute model updates"""
        if self.x is None:
            raise ValueError(f"Client {self.id}: Global model not received from server")
        
        if not self.data or len(self.data) == 0:
            # No local data - create zero updates
            self.delta_y = [torch.zeros_like(param.data, device=self.device) 
                           for param in self.x.parameters()]
            return
        
        # Create local copy of global model
        self.y = deepcopy(self.x).to(self.device)
        self.y.train()  # Set to training mode
        
        # Store initial parameters for computing delta
        initial_params = [param.data.clone() for param in self.y.parameters()]
        
        # Local training loop
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (inputs, labels) in enumerate(self.data):
                try:
                    # Move data to device and ensure correct types
                    inputs = inputs.float().to(self.device)
                    labels = labels.long().to(self.device)
                    
                    # Forward pass
                    self.y.zero_grad()
                    outputs = self.y(inputs)
                    loss = self.criterion(outputs, labels)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Manual SGD update (since we're not using optimizer)
                    with torch.no_grad():
                        for param in self.y.parameters():
                            if param.grad is not None:
                                param.data.sub_(param.grad, alpha=self.lr)
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"Client {self.id}: Error in batch {batch_idx}, epoch {epoch}: {e}")
                    continue
            
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                if epoch % max(1, self.num_epochs // 5) == 0:  # Log every 20% of epochs
                    print(f"Client {self.id}: Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
        
        # Compute parameter updates (delta_y = y_final - x_initial)
        with torch.no_grad():
            self.delta_y = []
            for param_y, param_initial in zip(self.y.parameters(), initial_params):
                delta = param_y.data.detach() - param_initial.detach()
                self.delta_y.append(delta.to(self.device))
        
        print(f"Client {self.id}: Local training completed")
    
    def get_data_size(self):
        """Get the number of samples in local dataset"""
        if not self.data:
            return 0
        
        total_samples = 0
        for batch in self.data:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                total_samples += len(batch[0])  # Assuming batch[0] contains inputs
            else:
                total_samples += 1  # Single sample
        return total_samples
    
    def evaluate_local(self):
        """Evaluate the local model on local data"""
        if self.y is None or not self.data:
            return 0.0, 0.0
        
        self.y.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.data:
                inputs = inputs.float().to(self.device)
                labels = labels.long().to(self.device)
                
                outputs = self.y(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(self.data) if len(self.data) > 0 else 0.0
        
        return avg_loss, accuracy