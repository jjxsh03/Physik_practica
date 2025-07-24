import os 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

# tensor conversion of dataset 
class Convert(torch.tensor):
    def __init__(self)
        

# set of modelling classes

'''
Linear fitting Model predicting optimal parameters for a predefined Epoch:
    fitting_data: x -> float
    device = gpu
'''

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
    
    def forward(self, x) -> float:
        return self.a * x + self.b


class ExpModel(torch.nn.Module):
    '''
    Model for fitting of data with an exponential trend:
        training epochs: 300 -> 1000
    '''
    def __init__(self):
        super(ExpModel, self).__init__()
        self.A = torch.nn.Parameter(torch.randn(()))
        self.B = torch.nn.Parameter(torch.tensor(0.1)) # gradiant descent updating 

    def forward(self, x) -> float:
        return self.A * torch.exp(self.B * x)

class CauchyModel(torch.nn.Module):
    '''
    General Fit for Lorentz/Cauchy Funtion
    '''
    def __init__(self):
        super(CauchyModel, self).__init__()
        self.A = torch.nn.Parameter(torch.randn(()))
        self.x_0 = torch.nn.Parameter(torch.randn(()))
        self.gamma = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        return self.A / (pow((x - self.x_0), 2) + pow(self.gamma, 2))

    def forward_normalized(self, x, *args):
        return self.gamma / (torch.pi * (pow((x - self.x_0), 2) + pow(self.gamma, 2)))



# Defining modelling workflow 
class ModelData():
    def __init__(self, model) -> int :
        self.model = model

    def train(self, device) -> int:

        






    
