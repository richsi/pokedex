import torch
import coremltools
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to model.pt")
args = vars(ap.parse_args())


