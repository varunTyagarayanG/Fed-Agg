import os
import json
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from mpi4py import MPI
from datetime import datetime
import importlib
from collections import OrderedDict
from src.util_functions import set_logger, save_plt, set_seed, evaluate_fn
from src.load_data_for_clients import dist_data_per_client
from src.models import CNN_Cifar, CNN_Mnist

class Client:
    def __init__(self, client_id, model, client_loader, fed_config, global_config):
        self.client_id = client_id
        self.model = model
        self.client_loader = client_loader
        self.fed_config = fed_config
        self.global_config = global_config
        self.device = global_config['device']
        
        # Use Adam optimizer for better convergence
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.fed_config['local_stepsize'])
        self.criterion = nn.CrossEntropyLoss()

    def train_local(self):
        self.model.train()
        for epoch in range(self.fed_config['num_epochs']):
            for images, labels in self.client_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return self.model.state_dict()


class Server:
    def __init__(self, model_config, global_config, data_config, fed_config, comm, rank, size):
        set_seed(global_config['seed'])
        
        self.comm = comm
        self.rank = rank
        self.size = size

        self.model_config = model_config
        self.global_config = global_config
        self.data_config = data_config
        self.fed_config = fed_config

        self.num_clients = fed_config['num_clients']
        self.num_rounds = fed_config['num_rounds']
        self.fraction_clients = fed_config['fraction_clients']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_config['device'] = self.device
        logging.info(f"Using device: {self.device}")

        # Initialize global model
        if self.model_config['name'] == 'CNN_Cifar':
            self.model = CNN_Cifar().to(self.device)
        elif self.model_config['name'] == 'CNN_Mnist':
            self.model = CNN_Mnist().to(self.device)
        else:
            raise ValueError("Model not supported")

        # Prepare data
        self.client_loaders, self.test_loader = dist_data_per_client(
            data_path=self.data_config['dataset_path'],
            dataset_name=self.data_config['dataset_name'],
            num_clients=self.num_clients,
            batch_size=self.fed_config['batch_size'],
            non_iid_per=self.data_config['non_iid_per'],
            device=self.device
        )
        self.criterion = nn.CrossEntropyLoss()

        self.results = {'accuracy': [], 'loss': []}

    def select_clients(self):
        num_selected = max(1, int(self.fraction_clients * self.num_clients))
        return self.comm.sample(range(self.num_clients), num_selected, replace=False)

    def train(self):
        logging.info("Starting federated training...")
        for round_num in range(self.num_rounds):
            logging.info(f"--- Communication Round {round_num + 1}/{self.num_rounds} ---")
            
            # Broadcast global model to all clients
            global_model_state = self.model.state_dict()
            global_model_state = self.comm.bcast(global_model_state, root=0)

            # Local training on clients
            if self.rank < self.num_clients:
                client = Client(
                    client_id=self.rank,
                    model=self.model,
                    client_loader=self.client_loaders[self.rank],
                    fed_config=self.fed_config,
                    global_config=self.global_config
                )
                local_model_state = client.train_local()
                
            else:
                local_model_state = None

            # Gather updates from all clients
            updated_models = self.comm.gather(local_model_state, root=0)

            # Server aggregation (only on rank 0)
            if self.rank == 0:
                self.aggregate_models(updated_models)
                
                # Evaluate global model
                test_loss, test_acc = evaluate_fn(self.test_loader, self.model, self.criterion, self.device)
                self.results['accuracy'].append(test_acc)
                self.results['loss'].append(test_loss)
                logging.info(f"Round {round_num + 1} | Global Model Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

    def aggregate_models(self, updated_models):
        if not updated_models:
            return

        global_model_state = self.model.state_dict()
        
        # Initialize an empty state dict to hold aggregated weights
        aggregated_weights = OrderedDict()
        for key in global_model_state.keys():
            aggregated_weights[key] = torch.zeros_like(global_model_state[key])

        # Sum up weights from all clients
        for client_state in updated_models:
            if client_state is not None:
                for key in aggregated_weights.keys():
                    aggregated_weights[key] += client_state[key]

        # Average the weights
        for key in aggregated_weights.keys():
            aggregated_weights[key] /= len(updated_models)

        # Update the global model with averaged weights
        self.model.load_state_dict(aggregated_weights)


def run_fl(Server, global_config, data_config, fed_config, model_config, comm, rank, size, run_id):
    # Create log directories only on root
    if rank == 0:
        log_dir = f"./Logs/{fed_config['algorithm']}/{data_config['non_iid_per']}/{run_id}/"
        os.makedirs(log_dir, exist_ok=True)

    comm.Barrier()

    # Set logger per rank with unique run ID
    log_filename = f"./Logs/{fed_config['algorithm']}/{data_config['non_iid_per']}/{run_id}/log_rank{rank}.txt"
    set_logger(log_filename)
    logging.info(f"Process {rank} is initializing the server")

    # Initialize server
    server = Server(model_config, global_config, data_config, fed_config, comm=comm, rank=rank, size=size)
    logging.info(f"Process {rank}: Server is successfully initialized")

    # Setup clients and start training
    server.train()

    # Save plots on root only
    if rank == 0:
        save_plt(list(range(1, server.num_rounds + 1)), server.results['accuracy'],
                 "Communication Round", "Test Accuracy",
                 f"./Logs/{fed_config['algorithm']}/{data_config['non_iid_per']}/{run_id}/accgraph.png")
        save_plt(list(range(1, server.num_rounds + 1)), server.results['loss'],
                 "Communication Round", "Test Loss",
                 f"./Logs/{fed_config['algorithm']}/{data_config['non_iid_per']}/{run_id}/lossgraph.png")
        logging.info("Plots saved successfully")

    logging.info(f"Process {rank}: Execution has completed")


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Load config on root
    if rank == 0:
        with open('config.json', 'r') as f:
            config = json.load(f)
    else:
        config = None

    # Broadcast config to all ranks
    config = comm.bcast(config, root=0)

    global_config = config["global_config"]
    data_config = config["data_config"]
    fed_config = config["fed_config"]
    model_config = config["model_config"]

    # Generate unique run ID using timestamp
    if rank == 0:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        run_id = None
    run_id = comm.bcast(run_id, root=0)

    # Dynamically import Server
    module_name = f"src.algorithms.{fed_config['algorithm']}.server"
    server_module = importlib.import_module(module_name)
    Server = server_module.Server

    # Run federated learning with the unique run ID
    run_fl(Server, global_config, data_config, fed_config, model_config, comm, rank, size, run_id)