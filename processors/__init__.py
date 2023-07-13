import torch
from torch import nn


class MNISTProcessor(nn.Module):
    """Processor for the MNIST dataset
    """
    def __init__(self):
        super().__init__()
    
    def preprocess(self, x):
        x = x.reshape(-1)
        return x.unsqueeze(0)
    

    def postprocess(self, x):
        x = x.reshape(28, 28, 1)
        return x


class FreyFaceProcessor(nn.Module):
    """Processor for Frey Face dataset
    """
    def __init__(self):
        super().__init__()
    
    def preprocess(self, x):
        x = x.reshape(-1)
        if x.dtype == torch.uint8:
            x = x / 256
        
        return x.unsqueeze(0)
    
    def postprocess(self, x):
        x = x.reshape(28, 20, 1)
        return x

__all__ = [
    "MNISTProcessor",
    "FreyFaceProcessor"
]