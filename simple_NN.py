import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#At the heart of PyTorch 
#data loading utility is the torch.utils.data.DataLoader class. 
#It represents a Python iterable over a dataset, with support for

from torch.utils.data import DataLoader 


#MINST dataset

train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

#DataLoader

train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)