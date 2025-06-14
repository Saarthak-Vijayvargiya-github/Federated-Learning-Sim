import tomllib
from pathlib import Path
import numpy as np
import pandas as pd

class CreateClientData():
    """ A class which creates the client Data with different properties.

    Parameters
    ----------
    seed = A seed value for random generator

    Makes a folder named iterations which has same number of CSV files as number of iterations
    """

    _instance = None

    def __new__(cls, seed):
        if cls._instance is None:
            cls._instance = super(CreateClientData, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, seed):
        if hasattr(self, "_initialized") and self._initialized:
            return
        np.random.seed(seed)
        self.paths = []
        self.get_toml_data()
        self.create_csv()
        self._initialized = True
        
    def get_toml_data(self):
        pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        self.num_clients =  data["tool"]["flwr"]["federations"]["local-simulation"]["options"]["num-supernodes"]
        self.num_iter = data["tool"]["flwr"]["app"]["config"]["num-server-rounds"]

    def create_csv(self):
        output_dir = Path(__file__).parent / "iterations"
        output_dir.mkdir(exist_ok=True)  # Makes the directory if it doesn't exist

        for i in range(self.num_iter):
            battery_support = np.random.standard_normal(self.num_clients) < np.random.uniform(0.6, 1)
            processor_support = np.random.standard_normal(self.num_clients) < np.random.uniform(0.6, 1)
            usable = battery_support & processor_support

            df = pd.DataFrame([battery_support, processor_support, usable],  index = ["Battery", "Proc", "Usable"]).T
            output_path = output_dir / f"Iteration_{i}.csv"
            df.to_csv(output_path, index=False)
            self.paths.append(output_path)

    def getIterData(self,index):
        return self.paths[index]