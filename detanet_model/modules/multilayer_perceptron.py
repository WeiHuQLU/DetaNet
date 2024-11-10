from torch import nn
from e3nn import o3
from e3nn.nn import Activation
from .acts import activations
class MLP(nn.Module):
    '''Invariant MLP, which can only act on scalar features'''
    def __init__(self,size,act,bias=True,last_act=False,dropout=0.0):
        super(MLP,self).__init__()
        assert len(size)>1,'Multilayer perceptrons must be larger than one layer'

        mlp=[]
        for si in range(0,len(size)-1):

            l=nn.Linear(size[si], size[si + 1], bias=bias)
            nn.init.xavier_uniform_(l.weight)
            l.bias.data.fill_(0)
            mlp.append(l)
            activation = activations(act, num_features=size[si + 1])
            if last_act is False:
                if (si!=len(size)-2):

                    mlp.append(activation)
                    mlp.append(nn.Dropout(p=dropout))
            else:
                mlp.append(activation)
                mlp.append(nn.Dropout(p=dropout))
        self.mlp=nn.Sequential(*mlp)

    def forward(self,x):
        for f in self.mlp:
            x=f(x)
        return x

class Equivariant_Multilayer(nn.Module):
    '''Equivariant MLP, which can only act on all features'''
    def __init__(self,irreps_list,act,last_act=False):
        super(Equivariant_Multilayer, self).__init__()
        e_mlp=[]
        for ei in range(0,len(irreps_list)-1):
            acts = []
            irreps_in=irreps_list[ei]
            irreps_out=irreps_list[ei+1]
            l=o3.Linear(irreps_in=irreps_in,irreps_out=irreps_out)
            for irr in o3.Irreps(irreps_out):
                if o3.Irrep('0e') in irr:
                    activation = activations(act, num_features=irr.dim)
                    acts.append(activation)
                else:
                    acts.append(None)
            e_act=Activation(irreps_in=irreps_out,acts=acts)
            e_mlp.append(l)
            if last_act is False:
                if (ei != len(irreps_list) - 2):
                    e_mlp.append(e_act)
            else:
                e_mlp.append(e_act)
        self.e_mlp = nn.Sequential(*e_mlp)

    def forward(self,x):
        for f in self.e_mlp:
            x=f(x)
        return x