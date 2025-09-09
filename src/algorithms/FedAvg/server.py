import numpy as np
import logging
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
from .client import Client
from src.models import *
from src.load_data_for_clients import dist_data_per_client
from src.util_functions import set_seed, evaluate_fn
from mpi4py import MPI

class Server():
    def __init__(self, model_config={}, global_config={}, data_config={}, fed_config={}, optim_config={}, comm=None, rank=0, size=1):
        set_seed(global_config["seed"])
        self.device = global_config["device"]

        self.data_path = data_config["dataset_path"]
        self.dataset_name = data_config["dataset_name"]
        self.non_iid_per = data_config["non_iid_per"]

        self.fraction = fed_config["fraction_clients"]
        self.num_clients = fed_config["num_clients"]
        self.num_rounds = fed_config["num_rounds"]
        self.num_epochs = fed_config["num_epochs"]
        self.batch_size = fed_config["batch_size"]
        self.criterion = eval(fed_config["criterion"])()
        self.lr = fed_config["global_stepsize"]
        self.lr_l = fed_config["local_stepsize"]

        self.x = eval(model_config["name"])()
        self.clients = None

        # MPI attributes
        self.comm = comm
        self.rank = rank
        self.size = size

    def create_clients(self, local_datasets):
        clients = []
        for id_num, dataset in enumerate(local_datasets):
            client = Client(client_id=id_num, local_data=dataset, device=self.device,
                            num_epochs=self.num_epochs, criterion=self.criterion, lr=self.lr_l)
            clients.append(client)
        return clients

    def setup(self, **init_kwargs):
        """Initializes all the Clients and splits the train dataset among them"""
        if self.rank == 0:
            local_datasets, test_dataset = dist_data_per_client(
                self.data_path, self.dataset_name, self.num_clients,
                self.batch_size, self.non_iid_per, self.device
            )
        else:
            local_datasets = None
            test_dataset = None

        # Broadcast test dataset to all processes
        test_dataset = self.comm.bcast(test_dataset, root=0)
        self.data = test_dataset

        # Split local_datasets among processes
        if self.rank == 0:
            datasets_per_rank = np.array_split(local_datasets, self.size)
        else:
            datasets_per_rank = None

        # Scatter datasets to all ranks
        local_dataset = self.comm.scatter(datasets_per_rank, root=0)

        self.clients = self.create_clients(local_dataset)
        logging.info(f"Process {self.rank}: Clients are successfully initialized")

    # --- Remaining methods unchanged ---
    # sample_clients, communicate, update_clients, server_update, step, train