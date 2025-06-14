from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr_datasets.visualization import plot_label_distributions
import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from fl_simulation.custom_strategy import DynamicFitFedAvg
from fl_simulation.client_data import CreateClientData
from fl_simulation.task import *

SEED = 0

def on_fit_config(server_round: int) -> Metrics:
    """Adjusts learning rate based on current round."""
    lr = 0.01
    # Apply a simple learning rate decay
    if server_round % 2 == 1:
        lr = lr/2
    return {"lr": lr}

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregates metrics from an evaluate round."""
    # Loop trough all metrics received compute accuracies x examples
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    # Return weighted average accuracy
    return {"accuracy": sum(accuracies) / total_examples}


def get_evaluate_fn(testloader, device):
    """Return a callback that evaluates the global model."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model using provided centralised testset."""

        net = Net()
        # Apply global_model parameters
        set_weights(net, parameters_ndarrays)
        
        net.to(device)
        loss, accuracy = test(net, testloader, device)

        return loss, {"cen_accuracy": accuracy}

    return evaluate

def get_fraction_fit(server_round : int):
    ccd = CreateClientData(SEED)
    df = pd.read_csv(ccd.getIterData(server_round-1))
    num_devices = sum(df["Usable"])
    return num_devices/(len(df))

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Initialize model parameters
    initial_model = Net()

    partition_id = context.run_config["num-supernodes"]
    num_partitions = context.run_config["num-supernodes"]+1

    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lr = 0.01

    train_loss = train(initial_model, trainloader, lr, local_epochs, device)
    _, val_accuracy = test(initial_model, valloader, device)

    ndarrays = get_weights(initial_model)
    parameters = ndarrays_to_parameters(ndarrays)

    print("Initial Training Loss:", train_loss)
    print("Initial Val Accuracy: ", val_accuracy)

    """Code For visualizing the Dataset."""
    # figure, _, _ = plot_label_distributions(
    #     partitioner = getFds().partitioners["train"],
    #     label_name = "label",
    #     plot_type = "heatmap",
    #     legend = True,
    #     plot_kwargs = {"annot": True}
    #     )
    # figure.tight_layout()
    # plt.show()
    
    # Define strategy
    strategy = DynamicFitFedAvg(
        fraction_fit_fn = get_fraction_fit,
        fraction_evaluate = 1.0,
        min_available_clients = 2,
        initial_parameters = parameters,
        evaluate_metrics_aggregation_fn = weighted_average,
        on_fit_config_fn = on_fit_config,
        evaluate_fn = get_evaluate_fn(valloader, device)
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
np.random.seed(SEED)
app = ServerApp(server_fn=server_fn)
