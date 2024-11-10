from torch import nn
import torch
from torch.nn import functional as F


class Swish(nn.Module):
    '''Sigmoid Linear Unit(SiLU) activation function with weights(also known as swish),
    from SpookyNet:https://doi.org/10.1038/s41467-021-27504-0'''
    def __init__(self,num_features,inital_alpha=1.00,inital_beta=1.702):
        super(Swish, self).__init__()
        self.initial_alpha = inital_alpha
        self.initial_beta = inital_beta
        self.register_parameter("alpha", nn.Parameter(torch.Tensor(num_features)))
        self.register_parameter("beta", nn.Parameter(torch.Tensor(num_features)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.alpha, self.initial_alpha)
        nn.init.constant_(self.beta, self.initial_beta)

    def forward(self,x):
        return x*self.alpha*torch.sigmoid(self.beta*x)




class HardSwish(nn.Module):
    '''hardswish activation'''
    def __init__(self,inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

class ShiftedSoftplus(nn.Module):
    '''ShiftedSoftplus activation from SchNet:https://doi.org/10.1063/1.5019779'''
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()
    def forward(self, x):
        return F.softplus(x) - self.shift

def activations(type,num_features=128):
    '''get activations'''
    act = {'swish': Swish(num_features=num_features),
       'hardswish': HardSwish(inplace=False),
       'shiftedsoftplus': ShiftedSoftplus(),
        'softmax':nn.Softmax(dim=-1),
        'silu':nn.SiLU(),
        'relu':nn.ReLU(),
        'sigmoid':nn.Sigmoid(),
        'tanh':nn.Tanh()
        }
    return act[type]