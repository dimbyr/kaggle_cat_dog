from tqdm import tqdm
import pip, torch, argparse
import numpy as np
import torch, os

depths  =[8, 16, 32, 48]
wsl = [f'resnext101_32x{i}d_wsl' for i in depths]
model_paths = dict(zip(depths, wsl))

def resnext101(depth = 32):
    md = torch.hub.load('facebookresearch/WSL-Images', model_paths[depth])
    md.fc = torch.nn.Linear(md.fc.in_features, 2)
    return md

