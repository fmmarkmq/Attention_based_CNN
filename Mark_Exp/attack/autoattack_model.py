import torch
import torch.nn as nn

class Model_for_Autoattack(nn.Module):
    def __init__(self, model, transform):
        super(Model_for_Autoattack, self).__init__()
        self.model = model
        self.transform = transform
    
    def forward(self, x):
        x = self.transform(x)
        x = self.model(x)
        return x