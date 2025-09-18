import torch
from copy import deepcopy
import logging
import numpy as np
from mpi4py import MPI

from .client import Client
from src.models import *
from src.load_data_for_clients import dist_data_per_client
from src.util_functions import set_seed, evaluate_fn

class Server:
    """
    MPI-compatible FedAdam server
    """
    def __init__(self, model_config={}, global_config={}, data_config={}, fed_config={}, comm=None, rank=None, size=None):
        set_seed(global_config["seed"])
        
        # MPI
        self.comm = comm if comm else MPI.COMM_WORLD
        self.rank = rank if rank is not None else self.comm.Get_rank()
        self.size = size if size is not None else self.comm.Get_size()

        self.device = global_config["device"]

        self.data_path = data_config["dataset_path"]
        self.dataset_name = data_config["dataset_name"]
        self.non_iid_per = data_config["non_iid_per"]

        self.num_clients = fed_config["num_clients"]
        self.num_rounds = fed_config["num_rounds"]
        self.num_epochs = fed_config["num_epochs"]
        self.batch_size = fed_config["batch_size"]
        self.fraction = fed_config["fraction_clients"]
        self.criterion = eval(fed_config["criterion"])()
        self.lr = fed_config["global_stepsize"]
        self.lr_l = fed_config["local_stepsize"]

        # Initialize model
        self.x = eval(model_config["name"])().to(self.device)
        
        # Initialize Adam state variables with proper initialization
        self.m = [torch.zeros_like(p.data, device=self.device, dtype=p.dtype) for p in self.x.parameters()]
        self.v = [torch.zeros_like(p.data, device=self.device, dtype=p.dtype) for p in self.x.parameters()]
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-6
        self.timestep = 1

        self.clients = []
        self.test_data = None

    def setup(self):
        """Split dataset across ranks and initialize clients"""
        if self.rank == 0:
            local_datasets, test_dataset = dist_data_per_client(
                self.data_path, self.dataset_name, self.num_clients,
                self.batch_size, self.non_iid_per, self.device
            )
            # Ensure we have enough datasets for all ranks
            if len(local_datasets) < self.size:
                # Pad with empty datasets if needed
                while len(local_datasets) < self.size:
                    local_datasets.append([])
            chunks = np.array_split(local_datasets, self.size)
        else:
            chunks = None
            test_dataset = None

        # Scatter datasets to all ranks
        local_data = self.comm.scatter(chunks, root=0)
        self.test_data = self.comm.bcast(test_dataset, root=0)

        # Initialize clients for this rank
        for idx, dataset in enumerate(local_data):
            if dataset is not None and len(dataset) > 0:  # Only create client if dataset exists
                client_id = self.rank * len(local_data) + idx
                self.clients.append(Client(
                    client_id=client_id,
                    local_data=dataset,
                    device=self.device,
                    num_epochs=self.num_epochs,
                    criterion=self.criterion,
                    lr=self.lr_l
                ))

        logging.info(f"Rank {self.rank}: Initialized {len(self.clients)} clients")

    def communicate(self):
        """Broadcast global model to all clients"""
        # Convert parameters to numpy arrays for MPI communication
        if self.rank == 0:
            x_params = [p.detach().cpu().numpy() for p in self.x.parameters()]
        else:
            x_params = None

        # Broadcast from rank 0 to all other ranks
        x_params = self.comm.bcast(x_params, root=0)

        # Update local model parameters
        for param, new_param in zip(self.x.parameters(), x_params):
            param.data = torch.tensor(new_param, device=self.device, dtype=param.dtype)

        # Update client models
        for client in self.clients:
            client.x = deepcopy(self.x)

    def update_clients(self):
        """Local client updates"""
        for client in self.clients:
            client.client_update()

    def server_update(self):
        """Aggregate delta_y from all clients and perform Adam update"""
        # Initialize local gradients
        local_grads = [torch.zeros_like(p.data, device=self.device, dtype=p.dtype) for p in self.x.parameters()]
        
        # Aggregate local client updates
        if len(self.clients) > 0:
            for client in self.clients:
                for g, delta in zip(local_grads, client.delta_y):
                    g.data += delta.data
            
            # Average by number of local clients
            for g in local_grads:
                g.data /= len(self.clients)

        # Aggregate across all ranks using MPI
        global_grads = []
        for g in local_grads:
            g_numpy = g.detach().cpu().numpy()
            total = np.zeros_like(g_numpy)
            self.comm.Allreduce(g_numpy, total, op=MPI.SUM)
            
            # Average by number of ranks
            total /= self.size
            global_grads.append(torch.tensor(total, device=self.device, dtype=g.dtype))

        # Apply Adam update
        self._apply_adam_update(global_grads)

    def _apply_adam_update(self, gradients):
        """Apply Adam optimizer update"""
        with torch.no_grad():
            for i, (p, g) in enumerate(zip(self.x.parameters(), gradients)):
                if g.numel() == 0:  # Skip empty gradients
                    continue
                    
                # Update biased first moment estimate
                self.m[i].mul_(self.beta1).add_(g, alpha=1 - self.beta1)
                
                # Update biased second raw moment estimate
                self.v[i].mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
                
                # Compute bias-corrected first moment estimate
                m_hat = self.m[i] / (1 - self.beta1**self.timestep)
                
                # Compute bias-corrected second raw moment estimate
                v_hat = self.v[i] / (1 - self.beta2**self.timestep)
                
                # Update parameters
                p.data.add_(m_hat / (torch.sqrt(v_hat) + self.epsilon), alpha=self.lr)

        self.timestep += 1

    def evaluate(self):
        """Evaluate the global model on test data"""
        if self.rank == 0 and self.test_data is not None:
            try:
                loss, acc = evaluate_fn(self.test_data, self.x, self.criterion, self.device)
                logging.info(f"Test Loss: {loss:.4f}, Accuracy: {acc:.2f}%")
            except Exception as e:
                logging.warning(f"Evaluation failed: {e}")

    def train(self):
        """Main training loop"""
        logging.info(f"Rank {self.rank}: Starting FedAdam training for {self.num_rounds} rounds")
        
        for round_idx in range(self.num_rounds):
            try:
                if self.rank == 0:
                    logging.info(f"Starting round {round_idx + 1}/{self.num_rounds}")
                
                # Synchronize all processes before each round
                self.comm.Barrier()
                
                # Broadcast current model to all clients
                self.communicate()
                
                # Perform local updates on clients
                self.update_clients()
                
                # Aggregate updates and perform server update
                self.server_update()
                
                # Synchronize before evaluation
                self.comm.Barrier()
                
                # Evaluate on rank 0
                if self.rank == 0:
                    self.evaluate()
                    logging.info(f"Completed round {round_idx + 1}/{self.num_rounds}")
                    
            except Exception as e:
                logging.error(f"Rank {self.rank}: Error in round {round_idx + 1}: {e}")
                raise e
        
        logging.info(f"Rank {self.rank}: Training completed")