import os
import yaml
import random
import numpy as np
import torch


def load_yaml(path):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_dir(path):
    if os.makedirs(name=path, exist_ok=True):
        print(f"Created {path} directory")


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
