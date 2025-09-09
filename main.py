import os
import json
import logging
import matplotlib.pyplot as plt
from mpi4py import MPI
from src.util_functions import set_logger, save_plt
import importlib


def run_fl(Server, global_config, data_config, fed_config, model_config, comm, rank, size):
    # Create directories only in the root process to avoid race conditions
    if rank == 0:
        log_dir = f"./Logs/{fed_config['algorithm']}/{data_config['non_iid_per']}/"
        os.makedirs(log_dir, exist_ok=True)

    # Synchronize all processes before continuing
    comm.Barrier()

    # Each process sets up its own logging file with rank distinction
    log_filename = f"./Logs/{fed_config['algorithm']}/{data_config['non_iid_per']}/log_rank{rank}.txt"
    set_logger(log_filename)

    logging.info(f"Process {rank} is initializing the server")

    # Pass comm, rank, and size to the Server
    server = Server(
        model_config,
        global_config,
        data_config,
        fed_config,
        comm=comm,
        rank=rank,
        size=size
    )

    logging.info(f"Process {rank}: Server is successfully initialized")

    server.setup()  # Initializes clients and splits data among processes
    server.train()  # Trains the global model for multiple rounds

    # Save plots only from the root process
    if rank == 0:
        save_plt(list(range(1, server.num_rounds + 1)), server.results['accuracy'],
                 "Communication Round", "Test Accuracy",
                 f"./Logs/{fed_config['algorithm']}/{data_config['non_iid_per']}/accgraph.png")
        save_plt(list(range(1, server.num_rounds + 1)), server.results['loss'],
                 "Communication Round", "Test Loss",
                 f"./Logs/{fed_config['algorithm']}/{data_config['non_iid_per']}/lossgraph.png")
        logging.info("Plots saved successfully")

    logging.info(f"Process {rank}: Execution has completed")


if __name__ == "__main__":
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Load config only in the root process
    if rank == 0:
        with open('config.json', 'r') as f:
            config = json.load(f)
    else:
        config = None

    # Broadcast config to all processes
    config = comm.bcast(config, root=0)

    global_config = config["global_config"]
    data_config = config["data_config"]
    fed_config = config["fed_config"]
    model_config = config["model_config"]

    # Dynamically import Server class
    module_name = f"src.algorithms.{fed_config['algorithm']}.server"
    server_module = importlib.import_module(module_name)
    Server = server_module.Server

    # Run federated learning with MPI
    run_fl(Server, global_config, data_config, fed_config, model_config, comm, rank, size)
