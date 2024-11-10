from .constant import *
from .model_loader import *
from torch_scatter import scatter
from torch_geometric.nn import radius_graph
def hessfreq(Hi,Hij,edge_index,masses,normal=False,linear=False,scale=0.965):
    '''Combine the Hessian matrices of the Hi and Hij parts and calculate the frequencies and normal coordinates

    normal:Whether or not to normalize the normal coordinates,Default: False

    linear:Is it a linear molecule? Default: False

    scale:Correction factor for frequency.Default:0.965

    It is worth mentioning that the output frequency may generate NaN in the first frequency,
    which corresponds to the negative frequency in quantum chemistry calculations, Simply delete this frequency.
    '''

    #Calculating the masses matrix
    wmasses=torch.pow(masses.to(Hi.device),-0.5).repeat_interleave(3, dim=0)
    wmat=wmasses[:,None]*wmasses[None,:]

    #Construction of hessian matrix by atomic part and interatomic part
    i, j = edge_index
    dia=torch.arange(0,len(Hi),step=1,device=Hi.device,dtype=int)
    hessian=torch.zeros(size=(Hi.shape[0],3,Hi.shape[0],3),device=Hi.device)
    hessian[j,:,i,:]=Hij
    hessian[dia,:,dia,:]=Hi
    hessian=hessian.reshape(Hi.shape[0]*3,Hi.shape[0]*3)

    #predicted hesssian matrix (ij part) may be asymmetric,we take the average of two directions to make it symmetric
    hessian=(hessian+hessian.permute(1,0))/2


    hessian=hessian*wmat

    #Diagonalisation
    eva, evec=torch.linalg.eigh(hessian)
    eva = eva * hess_t
    freq=torch.pow(eva,0.5)/ (2 * torch.pi)
    freq=freq/cm_hz
    p=-evec.t()*wmasses
    normals = torch.norm(p, dim=1).unsqueeze(1)
    if normal:
        p=p/normals
    p=p.reshape(len(freq),-1,3)
    if scale is not None:
        freq=freq*scale
    if linear:
        return freq[5:],p[5:]
    else:
        return freq[6:],p[6:]



def chain_rule_ir(dd,modes):
    '''Calculation of infrared intensity by the chain rule
    The input dipole moment derivatives unit are '(D/A)',
    output IR intensity unit are KM/Mol(Consistent with gaussian g16)
    '''
    irs=(modes[...,None]*dd[None,:,:,:]).reshape(modes.shape[0],-1,3)
    irxyz=torch.sum(irs,dim=1)
    ir=irxyz.norm(dim=-1)**2
    return ir*ir_coff

def chain_rule_raman(dp,modes):
    '''Calculation of raman tensor by the chain rule
    The input polarizability derivatives unit are 'a0**3/A',
    '''
    ramans=(modes[...,None]*dp[None,:,:,:]).reshape(modes.shape[0],-1,6)
    raman_tensor=torch.sum(ramans,dim=1)
    return raman_tensor

def get_raman_act(raman_tensor):
    '''Calculation of Raman activity from the Raman tensor
    calculate by 45x(alpha**2)+7x(gamma**2)  alpha and gamma represent isotropic and anisotropic part of Raman tensor
    The output Raman activity is in units of A**4/Amu(Consistent with gaussian g16)
    '''
    xx=raman_tensor[:,0]
    xy=raman_tensor[:,1]
    yy=raman_tensor[:,2]
    xz=raman_tensor[:,3]
    zy=raman_tensor[:,4]
    zz=raman_tensor[:,5]
    alpha=(xx+yy+zz)/3
    gamma_sq1=0.5*(((xx-yy)**2)+((yy-zz)**2)+((zz-xx)**2))
    gamma_sq2=3*((xy**2)+(xz**2)+(zy**2))
    gamma_sq=gamma_sq1+gamma_sq2
    return (45*(alpha**2)+7*gamma_sq)*raman_coff

def get_raman_intensity(freq,raman_act,temp=298,init_wl=532):
    '''Raman intensity is calculated from the Raman activity and frequency,temperature and incident light Wavelength.
     The output Raman intensity is unitless.
     Unfortunately, 32-float is not sufficient for Raman calculations and only 64-float can be used to calculate Raman
     '''
    init_freq_si=cm_hz/ (init_wl * 1e-7)
    freq_si=freq.to(torch.float64)*cm_hz
    act_si=raman_act.to(torch.float64)*(A_m**4)/45
    last=1/(1-torch.exp(-hp*c*freq_si/(Kb*temp)))
    dv=((init_freq_si-freq_si)**4)/freq_si
    return act_si*dv*last

class nn_vib_analysis(torch.nn.Module):
    '''Complete with IR and Raman simulations by coordinates and atomic types.
    Simulations can be carried out in batches.

    Unfortunately, the shape of a hessian matrix is [num_atom*3,num_atom*3]
    pytorch does not support turning hessian into a batch.
    We therefore used an iterative algorithm for diagonalising and calculating IR and Raman,
    with a small improvement in simulation time.
    But it is still fast, taking about 0.5s (gpu) and 1s (cpu) to simulate 1 molecule.
     simulate a batch of 16 molecules takes roughly 1s(gpu) and 4s(cpu)
    '''
    def __init__(self,device,Linear=False,scale=0.965):
        super(nn_vib_analysis,self).__init__()
        '''Linear:whether it is a linear molecule (linear molecule e.g. hydrocyanic acid and butadiene), 
        if it is linear then [3*num_atom-5] frequencies and intensities will be calculated instead of [3*num_atom-6]
        
        scale:correction factor, quantum chemistry calculations usually overestimated frequencies 
        and need to be corrected by a correction factor, the default is 0.965 
        (i.e. the correction factor of the reference data B3LYP/TZVP used for training)
        
        device:There are two options torch.device('cpu') and torch.device('cuda'), 
        representing the use of cpu or gpu for computation respectively
        
        '''

        self.device=device
        self.linear=Linear
        self.scale=scale
        self.model_Hi=Hi_model(device=device)
        self.model_Hij=Hij_model(device=device)
        self.model_dd=dedipole_model(device=device)
        self.model_dp=depolar_model(device=device)
    def forward(self,pos,z,edge_index=None,batch=None):
        pos=pos.to(self.device)
        z=z.to(self.device)
        if batch is not None:
            batch=batch.to(self.device)
        if edge_index is None:
            edge_index=radius_graph(x=pos,r=5.0,batch=batch)
        Hi= self.model_Hi(pos=pos,z=z,batch=batch)
        Hij= self.model_Hij(pos=pos,z=z,edge_index=edge_index,batch=batch)
        dd= self.model_dd(pos=pos,z=z,batch=batch)
        dp= self.model_dp(pos=pos,z=z,batch=batch)
        if batch is None:
            freq,modes=hessfreq(Hi=Hi,Hij=Hij,masses=atom_masses[z], edge_index=edge_index, normal=False
                            , linear=self.linear,scale=self.scale)
            ir_int=chain_rule_ir(dd=dd,modes=modes)
            raman_act=get_raman_act(chain_rule_raman(dp=dp,modes=modes))
            return freq,ir_int,raman_act
        else:
            vib_list = []
            atom_num=0
            for n in range(0,batch.max()):
                bat_edge_index_mid = edge_index[:,edge_index[1]>=atom_num]
                Hij_m=Hij[edge_index[1]>=atom_num]
                atom_num=atom_num+len(batch[batch==n])
                bat_edge_index = bat_edge_index_mid[:,bat_edge_index_mid[1] < atom_num]
                Hij_o = Hij_m[bat_edge_index_mid[1] < atom_num]
                freq, modes = hessfreq(Hi=Hi[batch==n], Hij=Hij_o,
                                       masses=atom_masses[z[batch==n]], edge_index=bat_edge_index-bat_edge_index.min()
                                       , normal=False, linear=self.linear, scale=self.scale)
                ir_int = chain_rule_ir(dd=dd[batch==n], modes=modes, sers=self.sers)
                raman_act = get_raman_act(chain_rule_raman(dp=dp[batch==n], modes=modes), sers=self.sers)
                vib_list.append([freq,ir_int, raman_act])
            return vib_list

class nmr_calculator(torch.nn.Module):
    def __init__(self,device,refc=187.653,refh=31.751):
        super(nmr_calculator, self).__init__()
        '''refc and refh represent the reference shielding isotropic fraction of tetramethylsilane 
        (under the same calculation conditions as the reference data) '''
        self.device=device
        self.nmrc_model=nmr_model(device,params='trained_param/qm9nmr/shield_iso_c.pth')
        self.nmrh_model=nmr_model(device,params='trained_param/qm9nmr/shield_iso_h.pth')
        self.refc=refc
        self.refh=refh

    def forward(self,pos,z,batch=None):
        sc=self.nmrc_model(z=z,pos=pos,batch=batch)[z==6]
        sh=self.nmrh_model(z=z,pos=pos,batch=batch)[z==1]
        return -sc+self.refc,-sh+self.refh

def nmr_sca(nc,nh,indexc,indexh):
    '''Summation NMRH and NMRC for all identical chemical environments'''
    shiftc=scatter(src=nc,index=indexc,dim=-1,reduce='mean')
    shifth=scatter(src=nh,index=indexh,dim=-1,reduce='mean')
    intc=scatter(src=torch.ones_like(nc),index=indexc,dim=-1,reduce='sum')
    inth=scatter(src=torch.ones_like(nh),index=indexh,dim=-1,reduce='sum')
    return shiftc,intc,shifth,inth

def Gaussian_broadening(x0, y0,c = torch.linspace(0, 13.5, 270), sigma=0.5):
    '''Gaussian broadening for absorption spectroscopies
    x0 yo:Position of the horizontal and vertical axes to be broadening

    c:Position of the broadening horizontal axis and number of points

    sigma:spread half width

    The position of the vertical axis after the output has been broadened.
    '''
    lx = x0[:, :, None] - c[None, None, :]
    ci = sigma / 2 * ((2 * 0.69314718056) ** 0.5)
    ly = torch.exp(-0.5 * (lx ** 2) / (ci ** 2)) / (ci * 2 * 3.1415926)
    y = torch.sum(y0[:, :, None] * ly, dim=1)
    return  y.view(-1)

def Lorenz_broadening(x0, y0,c=torch.linspace(500, 4000, 3501), sigma=12):
    '''Lorenz broadening for Vibration and NMR spectroscopies
    x0 y0:Position of the horizontal and vertical axes to be broadening

    c:Position of the broadening horizontal axis and number of points

    sigma:spread half width

    The position of the vertical axis after the output has been broadened.
    '''
    lx= x0[:,None]-c[None,:]
    ly= (sigma/(2*3.1415926))/(lx**2 + 0.25*(sigma**2))
    y= torch.sum(y0[:,None]*ly,dim=0)
    return y.view(-1)