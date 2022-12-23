import torch

from train import trainer_pl

device = "cuda" if torch.cuda.is_available() else "cpu"


trainer_pl(device)
