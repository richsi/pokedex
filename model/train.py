# set matploblib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

import numpy as np
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to dataset")
ap.add_argument("-m", "--model", required=True, help="path to model")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to accuracy/loss plot")
args = vars(ap.parse_args())

EPOCHS = 100
INIT_LR = 1e-3
BATCH_SIZE = 32
IMAGE_DIM = (96,96,3)

data = []
labels = []

print("[INFO] loading images...")
image_paths = sorted(list(paths.list))

print(image_paths)