import numpy as np
import logging
import torch
from copy import deepcopy
from mpi4py import MPI

from .client import Client
from src.models import *
from src.load_data_for_clients import dist_data_per_client
from src.util_functions import set_seed, evaluate_fn


class Server:
    """
    MPI-compatible MimeLite Server
    """

    def __init__(self, model_config={}, global_config={}, data_config={}, fed_config={}, optim_config={}, comm=None, rank=None, size=None):
        set_seed(global_config["seed"])

        # MPI setup
        self.comm = comm if comm else MPI.COMM_WORLD
        self.rank = rank if rank is not None else self.comm.Get_rank()
        self.size = size if size is not None else self.comm.Get_size()

        # Configs
        self.device = torch.device(global_config.get("device", "cpu"))
        self.data_path = data_config["dataset_path"]
        self.dataset_name = data_config["dataset_name"]
        self.non_iid_per = data_config["non_iid_per"]

        self.fraction = fed_config["fraction_clients"]
        self.num_clients = fed_config["num_clients"]
        self.num_rounds = fed_config["num_rounds"]
        self.num_epochs = fed_config["num_epochs"]
        self.batch_size = fed_config["batch_size"]
        self.criterion = eval(fed_config["criterion"])()
        self.lr = fed_config["global_stepsize"]     # server/global stepsize
        self.lr_l = fed_config["local_stepsize"]    # client/local stepsize
        self.beta = 0.9                             # MimeLite momentum param

        # Global model
        self.x = eval(model_config["name"])().to(self.device)
        self.state = [torch.zeros_like(p, device=self.device) for p in self.x.parameters()]

        self.clients = None
        self.data = None

    def setup(self):
        """Split data among clients on each rank"""
        local_datasets, test_dataset = dist_data_per_client(
            self.data_path,
            self.dataset_name,
            self.num_clients,
            self.batch_size,
            self.non_iid_per,
            self.device
        )
        self.data = test_dataset
        self.clients = [
            Client(client_id=i,
                   local_data=dataset,
                   device=self.device,
                   num_epochs=self.num_epochs,
                   criterion=self.criterion,
                   lr=self.lr_l)
            for i, dataset in enumerate(local_datasets)
        ]
        logging.info(f"Rank {self.rank}: Clients initialized")

    def sample_clients(self):
        """Select fraction of clients"""
        num_sampled = max(int(self.fraction * self.num_clients), 1)
        return sorted(np.random.choice(self.num_clients, num_sampled, replace=False).tolist())

    def communicate(self, client_ids):
        """Send global model x and state to sampled clients"""
        for idx in client_ids:
            self.clients[idx].x = deepcopy(self.x).cpu()
            self.clients[idx].state = deepcopy(self.state)

    def update_clients(self, client_ids):
        """Each client performs client_update"""
        for idx in client_ids:
            self.clients[idx].client_update()

    def server_update(self, client_ids):
        """Aggregate client updates using MimeLite aggregation"""
        self.x.to(self.device)

        # buffers for delta_y (client model updates)
        sum_delta_y = [torch.zeros_like(p, device="cpu") for p in self.x.parameters()]

        # ----- Local aggregation on this rank -----
        with torch.no_grad():
            for idx in client_ids:
                for sdy, delta in zip(sum_delta_y, self.clients[idx].delta_y):
                    sdy.add_(delta)

        # ----- MPI Allreduce: sum across all ranks -----
        for i in range(len(sum_delta_y)):
            buf = sum_delta_y[i].numpy(force=True)
            global_sum = np.zeros_like(buf)
            self.comm.Allreduce(buf, global_sum, op=MPI.SUM)
            sum_delta_y[i] = torch.from_numpy(global_sum)

        # total number of sampled clients across all ranks
        local_num = len(client_ids)
        total_num = self.comm.allreduce(local_num, op=MPI.SUM)

        # ----- MimeLite server update -----
        with torch.no_grad():
            # average delta_y
            avg_delta_y = [sdy / total_num for sdy in sum_delta_y]

            # momentum update of state
            for s, delta in zip(self.state, avg_delta_y):
                s.data = (1 - self.beta) * delta.data.to(self.device) + self.beta * s.data.to(self.device)

            # update global model x = x + η * state
            for param, s in zip(self.x.parameters(), self.state):
                param.data = param.data + self.lr * s.data

    def step(self):
        """Performs a single communication round"""
        sampled_client_ids = self.sample_clients()
        self.communicate(sampled_client_ids)
        self.update_clients(sampled_client_ids)
        logging.info(f"Rank {self.rank}: client_update completed")
        self.server_update(sampled_client_ids)
        logging.info(f"Rank {self.rank}: server_update completed")

    def train(self):
        """Perform multiple rounds of training"""
        self.results = {"loss": [], "accuracy": []}
        for round_idx in range(self.num_rounds):
            logging.info(f"Rank {self.rank}: Communication Round {round_idx + 1}")
            self.step()

            # Only rank 0 evaluates to avoid duplication
            if self.rank == 0:
                test_loss, test_acc = evaluate_fn(self.data, self.x, self.criterion, self.device)
                self.results["loss"].append(test_loss)
                self.results["accuracy"].append(test_acc)
                logging.info(f"Round {round_idx + 1}: Loss={test_loss:.4f}, Accuracy={test_acc:.2f}%")
