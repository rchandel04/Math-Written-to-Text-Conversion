import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from pathlib import Path
import torchvision.models as models
from PIL import Image

# hyperparameters
BATCH_SIZE = 8
NUM_WORKERS = os.cpu_count()
NUM_EPOCHS = 5
# setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# setup training and testing directories
god_dir = Path("data/")
train_dir = god_dir / "train"
test_dir = god_dir / "test"

# define how we want to transform our images
img_transform = transforms.Compose([transforms.Resize(size=(44, 44)), 
                                    transforms.ToTensor(),
                                    transforms.Grayscale(num_output_channels=1)])

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
        return self.classifier(self.convolve_block_2(self.convolve_block_1(x)))

# create a copy of vgg16 model from scratch

# class vgg16(nn.Module):
#     def __init__(self, input_shape: int, output_shape: int) -> None:
        
#         super(vgg16, self).__init__()
        
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(input_shape, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU())
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(), 
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU())
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer5 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU())
#         self.layer6 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU())
#         self.layer7 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer8 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#         self.layer9 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#         self.layer10 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer11 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#         self.layer12 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#         self.layer13 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.fc = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(1*1*512, 4096),
#             nn.ReLU())
#         self.fc1 = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(4096, 4096),
#             nn.ReLU())
#         self.fc2= nn.Sequential(
#             nn.Linear(4096, output_shape))
        
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.layer5(out)
#         out = self.layer6(out)
#         out = self.layer7(out)
#         out = self.layer8(out)
#         out = self.layer9(out)
#         out = self.layer10(out)
#         out = self.layer11(out)
#         out = self.layer12(out)
#         out = self.layer13(out)
#         out = out.reshape(out.size(0), -1)
#         out = self.fc(out)
#         out = self.fc1(out)
#         out = self.fc2(out)
#         return out
# define training and testing loops

def train_step(model: torch.nn.Module,
          dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer):
    
    model.train()
    
    train_loss = 0
    train_acc = 0
    
    for batch, (X, y) in enumerate(dataloader):
        
        X = X.to(device)
        y = y.to(device)
        
        pred = model(X)
        
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        pred_class = torch.softmax(pred, dim=1)
        pred_class = torch.argmax(pred_class, dim=1)
        train_acc += (pred_class == y).sum().item()/len(pred)
    
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    
    model.eval()
    
    test_loss = 0
    test_acc = 0
    
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
        
            X = X.to(device)
            y = y.to(device)
            
            test_pred = model(X)
            
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()
            
            pred_class = test_pred.argmax(dim=1)
            test_acc += (pred_class == y).sum().item()/len(pred_class)
    
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    
    return test_loss, test_acc

# define train and test functions

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int):
    
    results = {"train_loss": [],
        "train_acc": [],
    }
    
    for epoch in range(epochs):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)

    return results

def test(model: torch.nn.Module,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module):
    
    test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn)
    
    print(
        f"test_loss: {test_loss:.4f} | "
        f"test_acc: {test_acc:.4f}"
    )
    
# function to save model

def save_model(model, model_name, save_folder='saved_models'):
    
    MODEL_PATH = Path(save_folder)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    
    MODEL_NAME = model_name
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    
    torch.save(obj=model.state_dict(), f = MODEL_SAVE_PATH)
    
# function to load model

def load_model(model, model_name, save_folder='saved_models'):
    
    MODEL_PATH = Path(save_folder)
    
    MODEL_NAME = model_name
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    
    print(f"Loading model from: {MODEL_SAVE_PATH}")
    
    model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
    
    return model

# function to train, test, and save the model

def train_test_save_model(model: torch.nn.Module,
                          train_dataloader: torch.utils.data.DataLoader,
                          test_dataloader: torch.utils.data.DataLoader,
                          optimizer: torch.optim.Optimizer,
                          loss_fn: torch.nn.Module,
                          epochs: int,
                          save: bool,
                          model_name: str = 'None',
                          saved_folder: str = 'saved_models'):
    
    # train the model
    print("Training the model:")
    model_results = train(model=model,
                        train_dataloader=train_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=epochs)
    
    # test the model
    print("Testing the model:")
    test(model=model,
         test_dataloader=test_dataloader,
         loss_fn=loss_fn)
    
    # save the model if needed
    if save:
        print("Saving the model:")
        save_model(model, model_name, saved_folder)

    return model_results
# instantiate models

model_TinyVGG_0 = TinyVGG(input_shape=1, hidden_units=8, output_shape=len(train_data.classes)).to(device)

# model_VGG16_0 = vgg16(input_shape=1, output_shape=len(train_data.classes)).to(device)

# define loss function
loss_fn = nn.CrossEntropyLoss()

# define optimizer
optimizer = torch.optim.Adam(params=model_TinyVGG_0.parameters(), lr=0.001)

# train the model
model_results = train_test_save_model(model=model_TinyVGG_0,
                                      train_dataloader=train_dataloader,
                                      test_dataloader=test_dataloader,
                                      optimizer=optimizer,
                                      loss_fn=loss_fn,
                                      epochs=NUM_EPOCHS,
                                      save=False,
                                      model_name='TinyVGG_0'
                                      )

# function to perform prediction on single image input

def classifyImage(model: torch.nn.Module,
                  image_path: str):
    # load image and convert it to a tensor
    img = Image.open(image_path)
    img_tensor = img_transform(img)
    img_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)
    
    # get the target label
    parent_folder = os.path.dirname(image_path)
    target_label = os.path.basename(parent_folder)
    
    # make the prediction
    model.eval()
    with torch.inference_mode():
        pred = model(img_tensor)
        
        pred_prob = torch.softmax(pred.squeeze(), dim=0)
        
        pred_idx = torch.argmax(pred_prob, dim=0)
        pred_label = train_data.classes[pred_idx]
        
    # print out target and pred
    
    print(f"Target was: {target_label}")
    print(f"Predicted: {pred_label}")
    
# load the model
load_model(model_TinyVGG_0, 'TinyVGG_0')

classifyImage(model_TinyVGG_0, 'data/test/gamma/exp352.jpg')