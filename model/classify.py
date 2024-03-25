import torch
import model
from torchvision.transforms import v2
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to model.pt")
ap.add_argument("-i", "--image", required=True, help="path to image")
args = vars(ap.parse_args())

# load image
image = cv2.imread(args["image"])
output = image.copy()

# Preprocessing
transform = v2.Compose([
    v2.Resize((96,96)),
    v2.ToTensor(),
    v2.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = transform(image)
image = image.unsqueeze(0) # Add batch dimension

# Load pretrained model
print("[INFO] loading network...")
model = torch.load(args["model"])
model.eval()

# Classifying
print("[INFO] classying image...")
with torch.no_grad():
    output = model(image)
