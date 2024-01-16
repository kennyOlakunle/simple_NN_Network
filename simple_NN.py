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

test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

#Define the fully connected neural network

class NeuralNet(nn.module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.Relu()
        self.l2 = nn.Linear(hidden_size, num_classes)