import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from pathlib import Path
from torchinfo import summary
# hyperparameters
BATCH_SIZE = 8
NUM_WORKERS = os.cpu_count()
# setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# setup training and testing directories
god_dir = Path("data/")
train_dir = god_dir / "train"
test_dir = god_dir / "test"

# define how we want to transform our images
img_transform = transforms.Compose([transforms.Resize(size=(45, 45)), transforms.ToTensor()])

# create datasets for training and testing
train_data = datasets.ImageFolder(root=train_dir, transform=img_transform, target_transform=None)
test_data = datasets.ImageFolder(root=test_dir, transform=img_transform, target_transform=None)

# turn datasets into dataloaders

train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

# create a sequential CNN  (TinyVGG) model class

class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        
        super().__init__()
        
        self.convolve_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.convolve_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*11*11, out_features=output_shape)
        )
        
    def forward(self, x: torch.Tensor):
        self.classifier(self.convolve_block_2(self.convolve_block_1(x)))
    
# instantiate model

model_TinyVGG_0 = TinyVGG(input_shape=1, hidden_units=8, output_shape=len(train_data.classes)).to(device)

summary(model_TinyVGG_0, input_size=[1, 1, 44, 44])