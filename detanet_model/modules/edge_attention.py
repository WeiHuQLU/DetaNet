from torch import nn
from .acts import activations

class Edge_Attention(nn.Module):
    '''Radial attention module'''
    def __init__(self,num_radial,num_features,act,head=8):
        super(Edge_Attention,self).__init__()
        self.head=head
        self.actq = activations(act,num_features=num_features)
        self.actk = activations(act, num_features=num_features)
        self.actv = activations(act, num_features=2*num_features)
        self.acta = activations(act, num_features=2*num_features)
        self.softmax=activations('softmax')
        self.lq=nn.Linear(num_features,num_features)
        self.lk=nn.Linear(num_features,num_features)
        self.lv=nn.Linear(num_features,2*num_features)
        self.la=nn.Linear(2*num_features,2*num_features)
        self.lrbf=nn.Linear(num_radial,num_features,bias=False)
        self.lkrbf=nn.Linear(num_features,num_features,bias=False)
        self.lvrbf=nn.Linear(num_features,2*num_features,bias=False)
        self.feature=num_features
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lq.weight)
        self.lq.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lk.weight)
        self.lk.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lv.weight)
        self.lv.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lrbf.weight)
        nn.init.xavier_uniform_(self.lkrbf.weight)
        nn.init.xavier_uniform_(self.lvrbf.weight)
        nn.init.xavier_uniform_(self.la.weight)
        self.la.bias.data.fill_(0)


    def resize(self,x):
        return x.reshape(x.shape[0],self.head,-1)

    def attention(self, Q, K, V):
        d = Q.shape[-1]
        dot = Q @ K.permute(0, 2, 1)
        A = self.softmax(dot / (d**0.5))
        return (A @ V).reshape(-1, 2 * self.feature)

    def forward(self,S,rbf,index):
        i,j=index
        #The invariant feature S through 3 different linear layers to obtain query,key and value node feature
        sq=self.actq(self.lq(S))
        sk=self.actk(self.lk(S))
        sv=self.actv(self.lv(S))
        #The radial baseline is passed through a weight matrix
        #and then through two different weight matrices to obtain the radial weights of key and value
        rbf=self.lrbf(rbf)
        rk=self.lkrbf(rbf)
        rv=self.lvrbf(rbf)
        q=sq[i]
        k=sk[j]*rk
        v=sv[j]*rv
        #Finally an attention mechanism is performed and then a linear layer is output.
        #q is the feature of atom i,and k and v are the products of the features of atom j and the radial weights.
        return self.acta(self.la(self.attention(Q=self.resize(q),K=self.resize(k),V=self.resize(v))))



