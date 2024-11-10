
from torch import nn
from .update import Update
from .message import Message
class Interaction_Block(nn.Module):
    '''Interaction layers, each consisting of a message layer and an update layer.'''
    def __init__(self,
                 num_features,
                 act,
                 head,
                 num_radial,
                 irreps_sh,
                 irreps_T,
                 dropout
                 ):
        super(Interaction_Block,self).__init__()
        self.message=Message(head=head,num_radial=num_radial,act=act,
                             num_features=num_features,irreps_sh=irreps_sh)
        self.update=Update(num_features=num_features,act=act,irreps_mout=self.message.tp.irreps_out,
                           irreps_T=irreps_T,dropout=dropout)

    def forward(self,S,T,rbf,sh,index):
        mijt,mijs=self.message(S=S,rbf=rbf,sh=sh,index=index)
        T,S=self.update(T=T,S=S,mijt=mijt,mijs=mijs,index=index)
        return S,T





