import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

#Discard Model

# Inputs:
    # Hand: 34x4
    # Meld: 34x4x4
    # Discard: 34x4x4

# Output:
    # 34: one-hot


# Meld model:

# Inputs:
    # Stolen Tile: 34
    # Rest of meld (in hand): 34x3
    # Hand: 34x4
    # Meld: 34x4x4
    # Discard: 34x4x4

# Output size:
    # 2 outputs: take and not take

# Choose highest probability


DISCARD_CHANNELS = 9
MELD_CHANNELS = 10


class MahjongNetwork(nn.Module):
    input_channels: int
    output_size: int
    seed: int

    def __init__(self, input_channels: int, output_size: int, seed: int):
        super(MahjongNetwork, self).__init__()
        self.input_channels = input_channels
        self.output_size = output_size
        self.seed = torch.manual_seed(seed)
        self.stack = nn.Sequential(
            nn.Conv2d(input_channels, 16, (3, 4)),
            nn.ReLU(),
            nn.Conv2d(16, 64, (32, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.stack(state)


