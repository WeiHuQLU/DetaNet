import torch
from e3nn import o3,io
from torch import nn,FloatTensor
from .constant import atom_masses
from torch_geometric.nn import radius_graph
from .modules import Interaction_Block,Embedding,Radial_Basis,MLP,Equivariant_Multilayer
from torch.autograd import grad
from torch_scatter import scatter
"""Deep equivariant tensor attention Network(DetaNet) graph neural network model"""

'''Irreducible Representations(irreps) of vectorial and tensorial feature
    In DetaNet, vectors and tensors are represented by Irreducible Representations(irreps).
    More carefully, the irreps feature consists of features of different degrees l, order m, and parity p
    
    degree determines rotation equivariant of feature,Represented by the non-negative integers l=0,1,2,3,... 
     each degree have(2l+1) equivariant elements(rotation order), in the case of a vector, which has l=1 and has 2x1+1=3 
     elements, i.e. x,y,z. The scalar l=0, on the other hand, has only 1 element,are invariant.l=2 is a traceless tensor
     and can be combined with a scalar to form a 2nd-cartesian tensor.
     
    Parity is represented by the strings o or e, i.e. p=1 and p=-1,for odd and even parity respectively. parity of a
     tensor in physics is related to the degree: p=-1**(l). We have also adopted this strategy in our model.
     
    The irreps tensor can be considered as an independent subspace of the Cartesian tensor.
     Each Cartesian tensor can be represented by an irreps tensor. For example, a 2nd-order Cartesian tensor can 
    consist of a 2e irrep tensor(tressless tensor) and a 0e scalar(as tress),
     and a 3rd-order Cartesian tensor can consist of an 1o vector and an 3o irrep tensor.
     The higher order 4th, 5th order, and more Cartesian tensor can also be represented by the irreps tensor
'''

class DetaNet(nn.Module):
    def __init__(self,num_features:int=128,
                 act:str='swish',
                 maxl:int=3,
                 num_block:int=3,
                 radial_type:str='trainable_bessel',
                 num_radial:int=32,
                 attention_head:int=8,
                 rc:float=5.0,
                 dropout:float=0.0,
                 use_cutoff:bool=False,
                 max_atomic_number:int=9,
                 atom_ref:FloatTensor=None,
                 scale:float=1.0,
                 scalar_outsize:int=1,
                 irreps_out:str or o3.Irreps=None,
                 summation:bool=True,
                 norm:bool=False,
                 out_type:str='scalar',
                 grad_type:str=None,
                 device:torch.device=torch.device('cuda')):
        """Parameter introduction
    num_features:The dimension of the scalar feature and irreps feature(each order m), set to 128 by default.

    act: non-linear activation function,default is 'learnable swish'

    maxl:Maximum degree of the feature, default is 3. When the input is 3, the invariant (scalar) feature with l=0
    and the equivariant (vectorial, tensorial) feature with l=1,2,3 will be enabled

    num_block: the number of interaction layer(i.e. message passing layers), default is 3 layers.

    radial_type: Types of radial functions, we have 4 built-in radial functions,
    namely 'gaussian', 'bessel', 'exp_gaussian', and 'trainable_bessel'. After testing, 'trainable_bessel'
    works best and is therefore chosen as the default

    num_radial: the number of radial basis, set to 16 for small or MD datasets, 32 for large datasets.Default is 32.

    attention_head: Relates to the shape of the attention matrix in the attention mechanism and can have a small effect
    on training speed and accuracy. Must be set to a number that is divisible by num_features, default is 8.

    rc:cut-off radius,DetaNet constructs the local chemical environment of an atom within a radius,only inter-atomic
    distance smaller than the cut-off radius will be counted.
    The default is 5.0Å, which should be adjusted with training reference data.

    dropout: the proportion of features that will be dropped as they pass through the neural network,
    default is 0, i.e. no dropout. It is best not to dropout, especially on large data sets.

    use_cutoff:True or False uses the cutoff function, the default is false. If you use the bessel function, it contains
    the cutoff function itself, so it has almost no effect. If True, it will use the cosine cutoff function on the
    radial basis before entering the neural network.

    max_atomic_number: The maximum atomic number in the dataset, e.g. QM9 contains CHONF, the maximum is F:9,
    QM7-X contains CHONSCl, the maximum is Cl:17.The default is 9.

    atom_ref: the reference data for a single atom, is a tensor of the shape [n_atoms], which is needed when
    predicting energy, since energy can be considered as the sum of atomic energy and atomization energy.

    scale: the scale to multiply the output by, defaults to 1, i.e. no multiplication. Used for converting units etc.

    scalar_outsize: output size of scalars (invariants), default is 1, directly the size of scalar properties when
    predicting scalar properties, depending on the situation when predicting vector and tensor properties (see below).

    irrep_out: vector and tensor of the output, consisting of a non-negative integer and a string with the letters
    'o','e'. The number represents the degree, the letter 'o' stands for odd parity and the letter 'e' stands
    for even parity. e.g. a vector of degree 1 with odd parity is written as '1o', an untraceable tensor of degree 2
    with even parity is written as '2e', or if both need to be output, as '1o+2e'. Scalars written as '0e' are split
    in a separate scalar_outsize module and cannot be placed in that module, otherwise the values of the output
    scalars would both be 0. default is None,i.e. no vectors or tensors are output.

    summation: whether the atomic properties are summed to obtain the molecular properties. False if it is for atomic
    properties such as charge, fill in True for molecular properties. default is True.

    norm:When predicting vectors such as dipole moments, the sum of squares of the vectors is usually required.
    If this module is True, then the sum of squares of the result will be output. The default is False.

    out_type:The type of the output. We have 7 types of properties built into the model for the output function,
    the corresponding out_type are: scalar:scalar properties (e.g. energy).' dipole':dipole moment (electric dipole
    moment, magnetic dipole moment, transition dipole, etc.).' 2_tensor':2nd order tensor (e.g. Polarizability and
    quadrupole moments).' 3_tensor':3rd order tensor (e.g. first hyperpolarizablity).' R2':electronic spatial extent.
    'latent':direct output of scalar and tensor features after interacting layers. Other: direct output of
    scalar_outsize dimension and irrep tensor for irrep_out dimension. Default is 'scalar'

    grad_type:Type of derivative property, derivatives of 3 energies: 'force':atomic force, 'Hi':atomic part of Hessian
    matrix, 'Hij' interatomic part of Hessian matrix. For these 3 kinds, out_type must be 'scalar'.
    ' dipole','polar' are the derivatives of the dipole moment and polarizability respectively.
    For these 2 derivative properties, out_type must be 'dipole','2_tensor' respectively.

    device: device type, either torch.device('cuda') represents gpu or torch.device('cpu')  represents cpu
        """
        super(DetaNet,self).__init__()
        assert num_features%attention_head==0,'attention head must be divisible by the number of features'
        self.scale=scale
        self.ref=atom_ref
        self.norm=norm
        self.summation=summation
        self.scalar_outsize=scalar_outsize
        self.out_type=out_type
        self.rc=rc
        self.grad=grad_type
        # Generate the representation of irrep features.
        irreps_T = o3.Irreps((num_features, (l, (-1) ** l)) for l in range(1, maxl + 1))
        self.vdim = o3.Irreps(irreps_T).dim
        self.features=num_features
        self.T=num_block
        self.irreps_out=irreps_out
        # Generate the spherical harmonic function.
        irrs_sh=o3.Irreps.spherical_harmonics(lmax=maxl, p=-1)
        # Removal of scalars with l=0
        self.irreps_sh=irrs_sh[1:]
        self.Embedding=Embedding(num_features=num_features,act=act,device=device,max_atomic_number=max_atomic_number)
        self.Radial=Radial_Basis(radial_type=radial_type,num_radial=num_radial,use_cutoff=use_cutoff)
        blocks = []
        # interaction layers
        for _ in range(num_block):
            block=Interaction_Block(num_features=num_features,
                        act=act,
                        head=attention_head,
                        num_radial=num_radial,
                        irreps_sh=self.irreps_sh,
                        irreps_T=irreps_T,
                        dropout=dropout
                        )
            blocks.append(block)
        self.blocks=nn.Sequential(*blocks)
        # generate output layer
        if irreps_out is not None:
            mid = []
            for _, (l, p) in o3.Irreps(irreps_out):
                mid.append((num_features, (l, p)))
            irreps_mid = o3.Irreps(mid)
            self.tout=Equivariant_Multilayer(irreps_list=[irreps_T,irreps_mid,irreps_out],act=act)
        if scalar_outsize !=0:
            self.sout=MLP(size=(num_features,num_features,scalar_outsize),act=act,dropout=dropout)

        self.mass=atom_masses.to(device)
        #Module for conversion from irrep tensor to Cartesian tensor
        if out_type == '2_tensor':
            self.ct = io.CartesianTensor('ij=ji')

        elif out_type == '3_tensor':
            self.ct = io.CartesianTensor("ijk=jik=ikj")

        #Taking 6 matrix elements of the polarizability tensor (3x3) (9 of which 3 are symmetric, so there are 6).
        # Used to reduce calculation cost when calculating the derivative of polarizability
        if self.grad=='polar':
            self.mask=torch.tril(torch.ones(size=(3,3)),diagonal=0).flatten()

    def centroid_coordinate(self, z, pos, batch):
        '''Calculate the centre-of-mass coordinates of each atom.'''
        mass = self.mass[z].view(-1, 1)
        if batch is None:
            c = torch.sum(pos * mass, dim=0) / torch.sum(mass, dim=0)
            ra = (pos - c)
        else:
            c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
            ra = (pos - c[batch])
        return ra

    def cal_dipole(self,z,pos,batch,outs,outt):
        '''
        Calculate the dipole moment with a scalar_outsize of 1,irreps_out of '1o' The dipole calculation by s*ra+t(1o),
         where t(1o) represents the local dipole created by the charge deviation,s represents the charge,ra is
         centre-of-mass coordinates s*ra is the integral dipole '''
        ra=self.centroid_coordinate(z=z,pos=pos,batch=batch)
        return outs*ra+outt

    def cal_p_tensor(self,z,pos,batch,outs,outt):
        '''
        The 2nd order tensor is calculated from a trace s(0) of the tensor and a traceless tensor.
         The formula for calculating the traceless tensor is: Y2e(ra)*s(1)+t(2e),
         where Y2e(ra) is the non-local part of the tensor and t(2e) represents the local part.
         s(0) and s(1) represent the two scalars of the output.Y2e(ra) represents the spherical harmonic function using
          2e for the centre-of-mass coordinates, which is mapped vector to tressless tensor with irrep of 2e.
          Calculating the 2nd-tensor requires scalar_outsize=2,irreps_out='2e' ,and out_type='2_tensor'
        '''
        sa,sb=torch.split(outs,dim=-1,split_size_or_sections=[1,1])
        ra=self.centroid_coordinate(z=z,pos=pos,batch=batch)
        ta=o3.spherical_harmonics(l=self.irreps_out,x=ra,normalize=False)*sa
        return self.ct.to_cartesian(torch.concat(tensors=(sb,outt+ta),dim=-1))

    def cal_R_sq(self,z,pos,batch,outs):
        '''
        Calculation of Electronic spatial extent R2=(s*|ra|)**2
        Requires sclar_outsize=1,irreps_out=None and out_type='R2'
        '''
        ra=self.centroid_coordinate(z=z,pos=pos,batch=batch)
        return ((torch.norm(ra, p=2, dim=-1, keepdim=True) ** 2) * outs).reshape(-1)

    def cal_3_p_tensor(self,z,pos,batch,outs,outt):
        '''
        The 3rd-order Cartesian tensor consists of a 1o and a 3o irrep tensor.
        where the 1o part is calculated by s(0)*ra+t(1o). Similarly, the 3o part is calculated by s(1)*Y3o(ra)+t(3o)
        s(0) and s(1) are the 2 scalars of the output. Y3o is a spherical harmonic function of 3o irrep that maps the
         center of mass coordinates ra to the irrep tensor of 3o. t(1o) and t(3o) are the 2 irrep outputs of the model.
        Calculating the 3rd-tensor requires scalar_outsize=2,irreps_out='1o+3o' ,and out_type='3_tensor'
        '''
        sa, sb = torch.split(outs, dim=-1, split_size_or_sections=[1, 1])
        ra=self.centroid_coordinate(z=z,pos=pos,batch=batch)
        ta=o3.spherical_harmonics(l='1o',x=ra,normalize=False)*sa
        tb=o3.spherical_harmonics(l='3o',x=ra,normalize=False)*sb
        return self.ct.to_cartesian(outt+torch.concat(tensors=(ta,tb),dim=-1))

    def grad_hess_ij(self, energy, posj, posi, create_graph=True):
        '''Calculating the inter-atomic part of hessian matrices.Find the cross-derivative for the coordinates
         of atom i and atom j that interact on the interaction layer.
         require out_type='scalar' and grad_type='Hij' and sclar_outsize=1 and irreps_out=None
         '''
        fj = -grad([torch.sum(energy)], [posj], create_graph=create_graph)[0]
        Hji = torch.zeros((fj.shape[0], 3, 3), device=fj.device)
        for i in range(3):
            gji = -grad([fj[:, i].sum()], [posi], create_graph=create_graph, retain_graph=True)[0]
            Hji[:, i] = gji
        return Hji

    def grad_hess_ii(self, energy, posa, posb, create_graph=True):
        '''Calculating the atomic part of hessian matrices.
        We divide the input coordinates into two parts, one for atom i and one for atom j. The Hessian atomic part
         is derived from the output scalar by taking the cross-derivative of these two pairs of coordinates.
         require out_type='scalar' and grad_type='Hi' and sclar_outsize=1 and irreps_out=None
        '''
        f = -grad([torch.sum(energy)], [posa], create_graph=create_graph)[0]
        Hii = torch.zeros((f.shape[0], 3, 3), device=f.device)
        for i in range(3):
            gii = -grad([f[:, i].sum()], [posb], create_graph=create_graph, retain_graph=True)[0]
            Hii[:, i] = gii
        return Hii

    def grad_force(self, energy, pos, create_graph=True):
        '''The atomic force is calculated by deriving the output scalar from the input coordinates.
        require out_type='scalar' and grad_type='force' and sclar_outsize=1 and irreps_out=None
        '''
        force=-grad([torch.sum(energy)], [pos], create_graph=create_graph)[0]
        return force

    def grad_dipole(self, dipole, pos):
        '''Calculating the derivative of the dipole moment with respect to the coordinates
        require out_type='dipole' and grad_type='dipole' and sclar_outsize=1 and irreps_out='1o'
        '''
        dedipole = torch.zeros(size=(pos.shape[0], 3, 3), device=pos.device)
        for i in range(0, 3):
            dedipole[:, :, i] = -grad([dipole[:, i].sum()], [pos], create_graph=True)[0]
        return dedipole

    def grad_polarzability(self, polars, pos):
        '''
        Calculating the derivative of the polarzability with respect to the coordinates
        require out_type='2_tensor' and grad_type='polar' and sclar_outsize=2 and irreps_out='2e'
        '''
        polars = polars.flatten(start_dim=1)[:, self.mask == 1]
        depolar = torch.zeros(size=(pos.shape[0], 3, 6), device=pos.device)
        for i in range(0, 6):
            depolar[:, :, i] = -grad([polars[:, i].sum()], [pos], create_graph=True)[0]
        return depolar

    def forward(self,
                z,
                pos,
                edge_index=None,
                batch=None):
        '''
        z:Atomic number, LongTensor of shape [num_atom]

        pos: Atomic coordinates,FloatTensor with shape [num_atom,3]

        edge_index:Index of edge, LongTensor of shape [2,num_edge], default is None, if None
         it will be automatically generated in the model according to the cutoff radius rc.

        batch:Indicates which molecule the atom belongs to, usually used during training.
         LongTensor for [num_atom] shape.
        '''

        #If the derivative property is predicted, need to set requires_grad=True
        if self.grad is not None:
            pos.requires_grad=True

        #Generate atomic pairs (i.e. edges) with distances less than the cut-off radius based on coordinates
        if edge_index is None:
            edge_index=radius_graph(x=pos,r=self.rc,batch=batch)

        #Embedding of atomic types into scalar features (via one-hot nuclear and electronic features)
        S=self.Embedding(z)
        T=torch.zeros(size=(S.shape[0],self.vdim),device=S.device,dtype=S.dtype)
        i,j=edge_index

        #If it is necessary to obtain the atomic part of the hessian, the 2 sets of coordinates
        # of the generated i and j atoms need to be recorded.
        if self.grad=='Hi':
            posa=pos.clone()
            posb=pos.clone()
            posj = posa[j]
            posi = posb[i]
        else:
            posi=pos[i]
            posj=pos[j]

        #From ri-rj we obtain the coordinate difference vector and sum the squares to obtain the interatomic distance.
        rij = posj - posi
        r=torch.norm(rij,dim=-1)

        #Generation of the irrep tensor from the normalised coordinate vector via the spherical harmonic function
        sh = o3.spherical_harmonics(l=self.irreps_sh, x=rij/(r.view(-1,1)), normalize=True, normalization="component")
        #Radial basis are generated from distances.
        rbf = self.Radial(r)

        #via interaction layers
        for block in self.blocks:
            S,T=block(S=S,T=T,sh=sh,rbf=rbf,index=edge_index)

        #Output of irrep tensor from equivariant linear layers
        if self.irreps_out is not None:
            outt=self.tout(T)

        # Output of scalar from invariant linear layers
        if self.scalar_outsize!=0:
            outs=self.sout(S)

        #via output function.
        if self.out_type=='scalar':
            out=outs

        elif self.out_type=='dipole':
            out=self.cal_dipole(z=z,pos=pos,batch=batch,outs=outs,outt=outt)

        elif self.out_type=='2_tensor':
            out=self.cal_p_tensor(z=z,pos=pos,batch=batch,outs=outs,outt=outt)

        elif self.out_type=='R2':
            out=self.cal_R_sq(z=z,pos=pos,batch=batch,outs=outs)

        elif self.out_type=='3_tensor':
            out=self.cal_3_p_tensor(z=z,pos=pos,batch=batch,outs=outs,outt=outt)

        elif self.out_type=='latent':
            out=S,T
        else:
            out=outs,outt

        if self.ref is not None:
            out=out+self.ref[z].to(self.device).reshape(-1,1)

        #Summation of atomic properties to obtain molecular properties
        if self.summation:
            if batch is not None:
                out = scatter(src=out, index=batch, dim=0)
            else:
                out = torch.sum(input=out, dim=0)

        #Calculating the derivative properties
        if self.grad=='force':
            out=self.grad_force(energy=out,pos=pos)

        elif self.grad=='dipole':
            out=self.grad_dipole(dipole=out.reshape(-1,3),pos=pos)

        elif self.grad=='polar':
            out=self.grad_polarzability(polars=out.reshape(-1,3,3),pos=pos)

        elif self.grad=='Hij':
            out=self.grad_hess_ij(energy=out,posj=posj,posi=posi)

        elif self.grad=='Hi':
            out=self.grad_hess_ii(energy=out,posa=posa,posb=posb)

        #Calculation of sums of squares of properties (dipole moments)
        if self.norm:
            out=out.norm(dim=-1,keepdim=False)

        #Multiply by a scale (for converting units)
        if self.scale is not None:
            out=out*self.scale
        return out



