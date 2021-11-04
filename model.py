import torch
import torch.nn as nn
class NeuralNetw(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetw, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) #first linear layer 
        self.l2 = nn.Linear(hidden_size, hidden_size) #second linear layer
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU() #this is pour activating function
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end, because later we applied the cross _entropy loss
        return out
