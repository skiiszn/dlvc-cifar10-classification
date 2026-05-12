import torch
import torch.nn as nn
from pathlib import Path

class DeepClassifier(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)
    

    def save(self, save_dir: Path, suffix=None):
        '''
        Saves the model, adds suffix to filename if given
        '''

        filename = "best_model.pth" if suffix is None else f"best_model_{suffix}.pth"
        path = save_dir / filename
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        '''
        Loads model from path
        Does not work with transfer model
        '''
        
        self.net.load_state_dict(torch.load(path, map_location='cpu'))
        print(f"Model loaded from {path}")