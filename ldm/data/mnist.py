import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from einops import rearrange

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

class MNIST(Dataset):
    def __init__(self):
        self.data = datasets.MNIST(root='./mnist', train=True, transform=transforms.ToTensor(), download=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx][0]
        data = {}
        data['image'] = rearrange(normalize_to_neg_one_to_one(img), 'c h w -> h w c')
        return data
        