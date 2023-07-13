import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision import transforms as T
from scipy.io import loadmat
from pathlib import Path

class MNISTDataset(Dataset):
    def __init__(self, root="./data"):
        super().__init__()
        self.root = Path(root)
        transform = T.Compose([T.ToTensor(), nn.Flatten()])
        self.data = MNIST(root=self.root, train=True, download=True, transform=transform)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index:int):
        return self.data[index][0].squeeze(0)
    

class FreyFaceDataset(Dataset):
    def __init__(self, root="./data/FreyFace"):
        super().__init__()
        self.root = Path(root)
        obj = loadmat(self.root / 'frey_rawface.mat')
        self.data = torch.tensor(obj['ff'].transpose(1, 0) / 256)
        self.data = self.data.float()
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return self.data[index]
    
__all__ = [
    "MNISTDataset",
    "FreyFaceDataset"
]