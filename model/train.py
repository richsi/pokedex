# set matploblib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

import numpy as np
from imutils import paths
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from model import SmallVGGNet

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to dataset")
ap.add_argument("-m", "--model", required=True, help="path to model")
ap.add_argument("-s", "--output", required=True, help="path to where model.pt should be saved")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to accuracy/loss plot")
args = vars(ap.parse_args())

EPOCHS = 100
INIT_LR = 1e-3
BATCH_SIZE = 32
IMAGE_DIM = (96,96,3)

data = []
labels = []
train_loss = []
train_accuracy = []
val_loss = []
val_accuracy = []

# Getting image paths
print("[INFO] loading images...")
image_paths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(image_paths)

# Set transformation
transformation = v2.Compose([
    v2.Resize((96,96)),
    v2.ToTensor(),
    v2.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = ImageFolder(root=args["dataset"], transform=transformation)

# Split train and test
num_train = int(len(dataset) * 0.8)
num_test = len(dataset) - num_train
train_dataset, test_dataset = random_split(dataset, [num_train, num_test])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = SmallVGGNet(IMAGE_DIM[0], IMAGE_DIM[1], IMAGE_DIM[2], len(dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=INIT_LR)
scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.1)

# Training and saving model
for epoch in range(EPOCHS):
    model.train()
    running_loss, running_corrects = 0.0, 0

    # Training
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    train_loss.append(epoch_loss)
    train_accuracy.append(epoch_acc)

    # Validation
    model.eval()
    with torch.no_grad():
        running_loss, running_corrects = 0.0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(test_loader.dataset)
        epoch_acc = running_corrects.double() / len(test_loader.dataset)
        val_loss.append(epoch_loss)
        val_accuracy.append(epoch_acc)

    print(f'[INFO] Epoch {epoch+1}/{EPOCHS}, '
          f'Train Loss: {train_loss[-1]:.4f}, Train Acc: {train_accuracy[-1]:.4f}, '
          f'Val Loss: {val_loss[-1]:.4f}, Val Acc: {val_accuracy[-1]:.4f}')


torch.save(model.state_dict(), args["output"])

# Plotting
plt.figure(figsize=(12, 6))

# Plot training and validation loss
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(args["plot"])