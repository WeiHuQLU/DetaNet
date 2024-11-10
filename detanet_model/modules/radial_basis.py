import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

def cosine_cutoff(r,rc):
    '''Traditional cosine cut-off function.'''
    zeros = torch.zeros_like(r)
    rc = torch.where(r < rc, r, zeros)
    out=torch.cos(r/rc)
    return torch.where(r<rc,out,zeros)

def softplus_inverse(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x + torch.log(-torch.expm1(-x))

class Bessel_Function(nn.Module):
    '''bessel radial basis, first proposed by DimeNet: arXiv:2003.03123v1 [cs.LG] 6 Mar 2020
    We have made certain changes to it,Added learnable weighting'''
    def __init__(self,num_radial,rc,weighting=True,inital_beta=2.0):
        super(Bessel_Function,self).__init__()
        assert rc !=0
        if weighting:
            self.register_parameter('alpha',
                                    nn.Parameter(torch.arange(0,num_radial,dtype=torch.float32).view(1,num_radial)))
            self.register_parameter('beta',nn.Parameter(torch.Tensor(1,num_radial)))
            nn.init.constant_(self.beta,inital_beta)

        else:
            self.register_buffer("alpha", torch.arange(1,num_radial+1,dtype=torch.float32).view(1,num_radial))
            self.beta = 1.0
        self.rc=rc
        self.prefactor=(2 /rc) ** 0.5

    def forward(self,r,cutoff=None):
        r = r.view(-1, 1)
        rbf = self.prefactor * torch.sin(self.alpha*torch.pi * r / self.rc) /(self.beta*r)
        if cutoff is not None:
            rbf=rbf*cutoff.view(-1,1)
        return rbf

class Exp_Gaussian_function(nn.Module):
    '''Learnable gaussian radial basis proposed by physnet:DOI: 10.1021/acs.jctc.9b00181'''
    def __init__(self,num_radial,init_alpha=0.95,no_inf=False,exp_weighting=False):
        super(Exp_Gaussian_function,self).__init__()
        self.init_alpha=init_alpha
        self.exp_weighting=exp_weighting
        if no_inf:
            self.register_buffer(
                "center",
                torch.linspace(1, 0, num_radial + 1, dtype=torch.float32)[:-1],
            )
            self.register_buffer(
                "width",
                torch.tensor(1.0 * (num_radial + 1), dtype=torch.float32),
            )
        else:
            self.register_buffer(
                "center", torch.linspace(1, 0, num_radial, dtype=torch.float32)
            )
            self.register_buffer(
                "width", torch.tensor(1.0 * num_radial, dtype=torch.float32)
            )
        self.register_parameter(
            "_alpha", nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self._alpha, softplus_inverse(self.init_alpha))

    def forward(self, r, cutoff=None):
        expalphar = torch.exp(-F.softplus(self._alpha) * r.view(-1, 1))
        rbf =torch.exp(-self.width * (expalphar - self.center) ** 2)
        if cutoff is not None:
            rbf=cutoff.view(-1,1)*rbf
        if self.exp_weighting:
            return rbf * expalphar
        else:
            return rbf

class Radial_Basis(nn.Module):
    '''Radial basis functions with 4 functions, and cosine cut-off radius'''
    def __init__(self,radial_type='bessel',cut_function=cosine_cutoff,num_radial=32,rc=5.0,use_cutoff=False):
        super(Radial_Basis,self).__init__()
        self.rc=rc
        self.radial_type=radial_type
        self.cut=cut_function
        self.use_cutoff = use_cutoff
        radials={'gaussian':Exp_Gaussian_function(num_radial=num_radial,exp_weighting=False),
                 'exp_gaussian':Exp_Gaussian_function(num_radial=num_radial,exp_weighting=True),
                 'bessel':Bessel_Function(num_radial=num_radial,rc=rc,weighting=False),
                 'trainable_bessel':Bessel_Function(num_radial=num_radial,rc=rc,weighting=True),
        }
        self.radial=radials[radial_type]

    def forward(self,r):
        if self.use_cutoff:
            cutoff=self.cut(r=r,rc=self.rc)
        else:
            cutoff=None
        return self.radial(r,cutoff)



