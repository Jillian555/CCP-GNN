import torch
import torch.nn as nn
import torch.nn.functional as F


###MLP with lienar output
class MLP(nn.Module):
    def __init__(self, input_dim, dim=[], batch_norm=False):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = len(dim)
        self.batch_norm = batch_norm
        if len(dim) < 1:
            raise ValueError("number of layers should be positive!")
        elif len(dim) == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, dim[0])
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, dim[0]))
            if self.batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(dim[0]))

            for layer in range(self.num_layers - 2):
                self.linears.append(nn.Linear(dim[layer], dim[layer + 1]))
                if self.batch_norm:
                    self.batch_norms.append(nn.BatchNorm1d((dim[layer+1])))
            self.linears.append(nn.Linear(dim[-2], dim[-1]))
            if self.batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(dim[-1]))

    def reset_parameters(self):
        if self.linear_or_not:
            self.linear.reset_parameters()
        else:
            for layer in self.linears:
                layer.reset_parameters()
            if self.batch_norm:
                for bn in self.batch_norms:
                    bn.reset_parameters()

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers-1):
                if self.batch_norm:
                    h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                else:
                    h = F.relu(self.linears[layer](h))
            if self.batch_norm:
                h = self.batch_norms[-1](self.linears[-1](h))
            else:
                h = self.linears[-1](h)
            return h