import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from tqdm import tqdm
import sys
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

# Download AFHQ dataset
afhq_dataset = load_dataset("huggan/AFHQ", split="train")

# Custom PyTorch Dataset Wrapper
class AFHQDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']  # PIL Image
        label = self.dataset[idx]['label']  # Already 0, 1, or 2

        if self.transform:
            image = self.transform(image)

        return image, label

    
transform = transforms.Compose(
            [Resize((128, 128)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

# Create dataset object
full_dataset = AFHQDataset(afhq_dataset, transform=transform)

# Prepare labels
labels = [full_dataset[i][1] for i in range(len(full_dataset))]  # List of labels

# Split indices with stratification
train_indices, test_indices = train_test_split(
    range(len(full_dataset)),
    test_size=0.2,
    stratify=labels,
    random_state=123
)

train_dataset = Subset(full_dataset, train_indices)
test_dataset = Subset(full_dataset, test_indices)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



class Logger(object):
    def __init__(self, filename="training_log.txt"):
        self.terminal = sys.stdout  # Original stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)  # Print to console
        self.log.write(message)       # Write to file

    def flush(self):
        pass

def evaluate(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    test_accuracy = 100.0 * correct / total
    print(f'>>> Test Accuracy: {test_accuracy:.2f}%\n')

def train(model, train_loader, test_loader, num_epochs=5, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Redirect stdout to Logger
    sys.stdout = Logger("training_log.txt")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]', leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            loop.set_postfix(loss=loss.item(), acc=100.0 * correct / total)

        train_accuracy = 100.0 * correct / total
        avg_loss = running_loss / len(train_loader)

        print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_loss:.4f} | Train Acc: {train_accuracy:.2f}%')

        # Evaluate after every epoch
        evaluate(model, test_loader)

    # After training, restore normal stdout
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal

# Instantiate your CNN model
model = Net()

# Call the training function
train(model, train_loader, test_loader, num_epochs=5)

torch.save(model.state_dict(), "afhq_cnn_model.pth")
