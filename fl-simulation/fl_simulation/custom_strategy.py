from typing import Callable
from flwr.server.strategy import FedAvg
from flwr.common import Parameters

class DynamicFitFedAvg(FedAvg):
    """Custom FedAvg strategy which selects the number of clients by their availablity"""

    def __init__(self, fraction_fit_fn: Callable[[int], float], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fraction_fit_fn = fraction_fit_fn

    # Dynamically update fraction_fit per round
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):        
        self.fraction_fit = self.fraction_fit_fn(server_round)
        return super().configure_fit(server_round, parameters, client_manager)