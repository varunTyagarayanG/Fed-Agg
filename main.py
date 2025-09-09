import os
import json
import logging
import matplotlib.pyplot as plt
from mpi4py import MPI
from src.util_functions import set_logger, save_plt

def run_fl(Server, global_config, data_config, fed_config, model_config):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Create directories only in the root process to avoid race conditions
    if rank == 0:
        if not os.path.exists(f"./Logs/{fed_config['algorithm']}"):
            os.makedirs(f"./Logs/{fed_config['algorithm']}", exist_ok=True)
        if not os.path.exists(f"./Logs/{fed_config['algorithm']}/{data_config['non_iid_per']}"):
            os.mkdir(f"./Logs/{fed_config['algorithm']}/{data_config['non_iid_per']}")

    # Synchronize all processes before continuing
    comm.Barrier()

    # Each process sets up its own logging file with rank distinction
    filename = f"./Logs/{fed_config['algorithm']}/{data_config['non_iid_per']}/"
    set_logger(f"{filename}log_rank{rank}.txt")

    logging.info(f"Process {rank} is initializing the server")

    server = Server(model_config, global_config, data_config, fed_config, comm, rank, size)

    logging.info(f"Process {rank}: Server is successfully initialized")

    server.setup()  # Initializes clients and splits data among processes

    server.train()  # Trains the global model for multiple rounds

    # Save plots only from the root process
    if rank == 0:
        save_plt(list(range(1, server.num_rounds + 1)), server.results['accuracy'],
                 "Communication Round", "Test Accuracy", f"{filename}accgraph.png")
        save_plt(list(range(1, server.num_rounds + 1)), server.results['loss'],
                 "Communication Round", "Test Loss", f"{filename}lossgraph.png")
        logging.info("Plots saved successfully")

    logging.info(f"Process {rank}: Execution has completed")


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Load config only in the root process and broadcast it
    if rank == 0:
        with open('config.json', 'r') as file_obj:
            config = json.load(file_obj)
    else:
        config = None

    # Broadcast config to all processes
    config = comm.bcast(config, root=0)

    global_config = config["global_config"]
    data_config = config["data_config"]
    fed_config = config["fed_config"]
    model_config = config["model_config"]

    # Dynamically import Server as in original code
    exec(f"from src.algorithms.{fed_config['algorithm']}.server import Server")

    # Pass comm, rank, size to the server if needed
    run_fl(Server, global_config, data_config, fed_config, model_config)
