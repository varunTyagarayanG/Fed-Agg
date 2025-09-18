import numpy as np
import torch
from copy import deepcopy
from mpi4py import MPI

from .client import Client
from src.models import *
from src.load_data_for_clients import dist_data_per_client
from src.util_functions import evaluate_fn

class Server:
    def __init__(self, global_config, data_config, fed_config, model_config, comm, rank, size):
        self.comm = comm
        self.rank = rank
        self.size = size

        self.device = global_config["device"]
        self.num_rounds = fed_config["num_rounds"]
        self.clients_per_round = fed_config["clients_per_round"]

        # Init global model
        self.global_model = init_model(model_config).to(self.device)

        # Setup clients (only rank 0 initializes full list)
        if self.rank == 0:
            client_data = dist_data_per_client(data_config)
            self.clients = [
                Client(cid, data, self.device, fed_config["num_epochs"], fed_config["lr"], torch.nn.CrossEntropyLoss())
                for cid, data in client_data.items()
            ]
        else:
            self.clients = None

    def broadcast_model(self):
        """Broadcast global model parameters to all clients"""
        weights = [p.data.cpu().numpy() for p in self.global_model.parameters()]
        weights = self.comm.bcast(weights, root=0)
        for p, w in zip(self.global_model.parameters(), weights):
            p.data = torch.tensor(w, dtype=p.data.dtype, device=self.device)

    def aggregate(self, deltas, grads):
        """MimeLite aggregation: combine deltas + gradients"""
        num_clients = len(deltas)
        avg_delta = [torch.zeros_like(p) for p in deltas[0]]
        avg_grad = [torch.zeros_like(p) for p in grads[0]]

        for d, g in zip(deltas, grads):
            for i in range(len(d)):
                avg_delta[i] += d[i] / num_clients
                avg_grad[i] += g[i] / num_clients

        # Update global model
        with torch.no_grad():
            for p, d, g in zip(self.global_model.parameters(), avg_delta, avg_grad):
                p.data -= 0.5 * (d + g)  # MimeLite update rule (simplified)

    def server_update(self, sampled_client_ids):
        """Collect updates from sampled clients and aggregate"""
        deltas, grads = [], []
        for cid in sampled_client_ids:
            delta, grad = self.clients[cid].client_update()
            deltas.append(delta)
            grads.append(grad)

        self.aggregate(deltas, grads)

    def step(self):
        """Run one training round"""
        if self.rank == 0:
            sampled_client_ids = np.random.choice(len(self.clients), self.clients_per_round, replace=False)
        else:
            sampled_client_ids = None
        sampled_client_ids = self.comm.bcast(sampled_client_ids, root=0)

        if self.rank == 0:
            self.server_update(sampled_client_ids)

        self.broadcast_model()

    def train(self):
        for r in range(self.num_rounds):
            if self.rank == 0:
                print(f"Round {r+1}/{self.num_rounds}")
            self.step()
