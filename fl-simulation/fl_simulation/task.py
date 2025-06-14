"""fl-simulation: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
import warnings
warnings.filterwarnings("ignore", message="The currently tested dataset are")

torch.manual_seed(123)

class Net(nn.Module):
    """Model of the CNN used for AFHQ Dataset"""

    def __init__(self):                             # Output sizes
        super(Net, self).__init__()                 # 3 x 128 x 128   [Img Size]
        self.conv1 = nn.Conv2d(3, 6, 5)             # 6 x 124 x 124
        self.pool = nn.MaxPool2d(2, 2)              # 6 x 62 x 62
        self.conv2 = nn.Conv2d(6, 16, 5)            # 16 x 58 x 58, Pool : 16 x 29 x 29
        self.conv3 = nn.Conv2d(16, 28, 3)           # 28 x 27 x 27, Pool : 28 x 13 x 13
        self.fc1 = nn.Linear(28 * 13 * 13, 512)
        self.fc2 = nn.Linear(512, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 28 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset

def get_transforms():
    """Apply transforms to the partition from FederatedDataset."""

    def apply_transforms(batch):
        pytorch_transforms = Compose(
            [Resize((128, 128)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        # print("\n\n\n", batch)
        # exit(0)
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch
    
    return apply_transforms

def load_data(partition_id: int, num_partitions: int):
    """Load partition AFHQ data."""

    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="label", alpha=10.0)
        fds = FederatedDataset(
            dataset="huggan/AFHQ",
            partitioners={"train": partitioner},
        )

    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    partition_train_test = partition_train_test.with_transform(get_transforms())
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)

    return trainloader, testloader


def train(net, trainloader, lr, epochs, device):
    """Train the model on the training set."""

    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def getFds():
    return fds