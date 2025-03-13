import os 
import sys
import h5py
import copy
import numpy as np
from time import time
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinSite
from tenpy.networks.site import Site
from tenpy.networks.site import SpinHalfFermionSite
from tenpy.algorithms import dmrg
from tenpy.tools.hdf5_io import save,load
from tenpy.models.tf_ising import TFIChain
from tenpy.models.lattice import Lattice
from tenpy.models.lattice import Chain
from tenpy.models.lattice import Lattice,get_order
from tenpy.models.model import CouplingMPOModel
from tenpy.models.model import CouplingModel, \
NearestNeighborModel, MPOModel

##############################################################################################
control_params = sys.argv[:]
print(control_params, flush=True)

model_params = { 
        'L': 2,                 # Length of the chain
        'bc_x':'periodic',      # Bound conditions of the lattice
        'bc_MPS':'infinite',    # Bound conditions of the MPS
        'mu':0,
        't1': 1.,               # Intracell nearest neighbor hopping (out of the diamond)
        't2': .5,               # Intracell next nearest neighbor coupling (inside the diamond)
        't2p': .5,              # Spin orbit coupling
        't3': 1.,               # Intercell coupling
        'U': 0.25               # Hubbard parameter
}
dmrg_params = {
        'mixer': True,               # setting this to True helps to escape local minima
        'mixer_params': {
            'amplitude': 1.e-5,      # amplitud del mixer
            'decay': 1.2,            # parametro que controla como de rapido decae la amplitud del mixer 
            'disable_after': 30      # desactivo el mixer despues de 20 sweeps
        },
        'trunc_params': {            # Controlamos algunas cosas del DMRG 
            'svd_min': 1.e-10,       # valor minimo de singular value permitido, los menore se truncan directamente
        },
        'lanczos_params': {          # Lanczos se ocupa de la mimizacion de la energia, es un algoritmo iterativo 
            'N_min': 5,              # numero minimo de autovalores a calcular
            'N_max': 20              # numero maximo de autovalores a calcular
        },
        'chi_list': {
            0: 9,
            10: 49,
            20: 258
        },
        'max_E_err': 1.e-10,         # Error maximo en la entropia
        'max_S_err': 1.e-4,          # Error maximo en la energia
        'max_sweeps': 100,           # Numero maximo de sweeps que realiza antes de terminar
        'max_hours': 0.5             # Maximo tiempo que corre el algoritmo en horas (0.00014 horas = 30 segundos)
}

T_params = { 't2_min': 0.0, 
            't2_max': 2.0, 
            't2_steps': 0.5, 
            't3_min': 0.0, 
            't3_max': 2.0, 
            't3_steps': 0.5}

if control_params[1] == 'R':
    Recycle = True
else:
    Recycle = False

#Actualizadores por linea de comandos Recycle / U /  T_max / t2_steps / T_min / chi_list
if len(control_params) > 2: 
    if control_params[2] != 'P':     
        model_params.update({'U': float(control_params[2])})
        print("U = ",model_params['U'],flush=True)

if len(control_params) > 3:
    if control_params[3] != 'P':
        T_params.update({'t2_max': float(control_params[3])})
        T_params.update({'t3_max': float(control_params[3])})
        print("T_max = ",T_params['t2_max'],flush=True)

if len(control_params) > 4: 
    if control_params[4] != 'P':
        T_params.update({'t2_steps': float(control_params[4])})
        T_params.update({'t3_steps': float(control_params[4])})        
        print("t2_steps = ",T_params['t2_steps'],flush=True)    

if len(control_params) > 5 : 
    if control_params[5] != 'P':
        dmrg_params.update({'t2_min': float(control_params[5])})
        dmrg_params.update({'t3_min': float(control_params[5])})
        print("T_min = ",dmrg_params['t2_min'],flush=True)

if len(control_params) > 6 :
    if control_params[6] != 'P':
        job_id = control_params[6]
        print("job_id = ",job_id,flush=True)

if len(control_params) > 7 :
    if control_params[7] != 'P':
        date = control_params[7]
        print("date = ",date,flush=True)

carpeta = f"/scratch/villodre/Hubbard_calculations/resultados/resultados_correlacion/{job_id}_{date}"

class DiamondChain(Lattice):
    dim = 1  #: the dimension of the lattice 
    Lu = 4  #: the (expected) number of sites in the unit cell, ``len(unit_cell)``.

    def __init__(self, L, site, **kwargs):
        sites = [site] * 4                               # Define the sites in the unit cell
        basis = np.eye(1)                                # La base es la trivial   
        pos = np.array([[0.2],[0.4],[0.6],[0.8]])        # Equiespaciados en el intervalo [0,1], no pongo nada ni en 0 ni en 1 para no superponerlos con la siguiente celda
        kwargs.setdefault('basis', basis)                # Guardo en el diccionario kwargs la base
        kwargs.setdefault('positions', pos)              # Guardo en el diccionario kwargs las posiciones
        NN = [(1, 0, np.array([0])), (0, 2, np.array([0])), (3, 1, np.array([0])),
             (2, 3, np.array([0]))]                     
        NNN = [(0,3,np.array([0])),(2,1,np.array([0]))]
        intercell = [(3,0,np.array([1]))]
        kwargs.setdefault('pairs', {})                   # Creo pairs para guardar todos los vecinos.
        kwargs['pairs'].setdefault('nearest_neighbors', NN)    
        kwargs['pairs'].setdefault('next_nearest_neighbors', NNN)
        kwargs['pairs'].setdefault('intercell', intercell)
        Lattice.__init__(self, [L], sites, **kwargs)     # Llamo al constructor de la clase Lattice, con 

    def ordering(self, order):
        if isinstance(order, str):
            if order == "default":
                priority = None
                snake_winding = (False, False, False)
                return get_order(self.shape, snake_winding, priority)
        return super().ordering(order)
    
class SpinfulHubbardDiamondChain(CouplingMPOModel):
    default_lattice = DiamondChain
    force_default_lattice = True
    def init_sites(self, model_params):
        cons_N = model_params.get('cons_N', 'N')
        cons_Sz = model_params.get('cons_Sz', 'Sz')
        site = SpinHalfFermionSite(cons_N=cons_N, cons_Sz=cons_Sz)
        return site

    def init_terms(self, model_params):
        # 0) Read out/set default parameters.
        t1 = model_params.get('t1', 1.)        
        t2 = model_params.get('t2', 1.)
        t3 = model_params.get('t3', 1.)
        U = model_params.get('U', 0)
        mu = model_params.get('mu', 0.)

        for u in range(len(self.lat.unit_cell)):  # El rango recorre el numero de sitios de la celda unitaria (4 en este caso)
            self.add_onsite(mu, u, 'Ntot')        # Añado un termino al hamiltoniano 
            self.add_onsite(U, u, 'NuNd')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:           # Añado los terminos de NN, los cuales he definido antes en la lattice
            self.add_coupling(-t1, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)
            self.add_coupling(-t1, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)
        for u1, u2, dx in self.lat.pairs['next_nearest_neighbors']:      # Añado los terminos de NNN, los cuales he definido antes en la lattice
            self.add_coupling(-t2, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)
            self.add_coupling(-t2, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)
        for u1, u2, dx in self.lat.pairs['intercell']:                   # Añado los terminos de intercell, los cuales he definido antes en la lattice
            self.add_coupling(-t3, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)
            self.add_coupling(-t3, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)

class SOCHubbardDiamondChain(CouplingMPOModel):
    default_lattice = DiamondChain
    force_default_lattice = True
    def init_sites(self, model_params):
        cons_N = model_params.get('cons_N', 'N')
        cons_Sz = model_params.get('cons_Sz', 'Sz')
        site = SpinHalfFermionSite(cons_N=cons_N, cons_Sz=cons_Sz)
        return site

    def init_terms(self, model_params):
        t1 = model_params.get('t1', 1.)
        t2 = model_params.get('t2', 1.)
        t2p = model_params.get('t2p',1.)
        t3 = model_params.get('t3', 1.)
        U = model_params.get('U', 0)
        mu = model_params.get('mu', 0.)

        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(mu, u, 'Ntot')
            self.add_onsite(U, u, 'NuNd')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-t1, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)
            self.add_coupling(-np.conj(t1), u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)

        # This is the regular term, similar to the one above 
        self.add_coupling(-t2,2,'Cdu',1,'Cu',np.array([0]),plus_hc=True)
        self.add_coupling(-t2,2,'Cdd',1,'Cd',np.array([0]),plus_hc=True)

        # This is the different term
        self.add_coupling(-t2p,0,'Cdu',3,'Cu',np.array([0]),plus_hc=True)
        self.add_coupling(-t2p,0,'Cdd',3,'Cd',np.array([0]),plus_hc=True)
        
        for u1, u2, dx in self.lat.pairs['intercell']:
            self.add_coupling(-t3, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)
            self.add_coupling(-t3, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)

class HubbardDiamondChain(CouplingMPOModel):
    default_lattice = DiamondChain
    force_default_lattice = True
    def init_sites(self, model_params):
        cons_N = model_params.get('cons_N', 'N')
        cons_Sz = model_params.get('cons_Sz', 'Sz')
        site = SpinHalfFermionSite(cons_N=cons_N, cons_Sz=cons_Sz)
        return site

    def init_terms(self, model_params):
        # 0) Read out/set default parameters.
        t1 = model_params.get('t1', 1.)
        t2 = model_params.get('t2', 1.)
        t3 = model_params.get('t3', 1.)
        U = model_params.get('U', 0)
        mu = model_params.get('mu', 0.)

        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(mu, u, 'Ntot')
            self.add_onsite(U, u, 'NuNd')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-t1, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)
            self.add_coupling(-t1, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)
        for u1, u2, dx in self.lat.pairs['next_nearest_neighbors']:
            self.add_coupling(-t2, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)
            self.add_coupling(-t2, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)
        for u1, u2, dx in self.lat.pairs['intercell']:
            self.add_coupling(-t3, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)
            self.add_coupling(-t3, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)    

##################################################################################################
def single_run(model_params,dmrg_params,prod_state=None,env_sweeps = 0,save_resume=True):
    # First, we incialize the product state, in order to have a generical one 
    if prod_state is None:
        prod_state = ["up","empty"] * (2 * model_params['L'])
    print(prod_state)

    # We create the product state
    M = SOCHubbardDiamondChain(model_params)            # M is the model
    # We create a MPS with the product state
    psi = MPS.from_product_state(M.lat.mps_sites(), prod_state, bc=M.lat.bc_MPS)    # MPS is the matrix product state
    print(psi.chi)
    # We create the DMRG engine
    eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    
    nexpvalup = psi.expectation_value("Nu")
    nexpvaldn = psi.expectation_value("Nd")
    print("Occupation up: ",np.average(nexpvalup))
    print("Occupation down: ",np.average(nexpvaldn))
    
    start_time = time()
    # We run the DMRG 
    if env_sweeps > 0:                              # If we have environment sweeps, we run them
        eng.environment_sweeps(env_sweeps)
    E, psi = eng.run()                              # We run the DMRG, obtaining the energy and the optimized MPS
    #print("DMRG done in ",(time()-start_time)/60," min")
    print("Energy = ",E)
    entropy = psi.entanglement_entropy()
    #print("Entanglement entropy:\n",entropy)
    print("correlation length = ",MPS.correlation_length(psi))
    print("Bond dimensions = ",psi.chi)
    if save_resume:
        save(eng.get_resume_data(),'resume_40.hdf5')

    return M,psi

def CorrelationPoint_R(model_params,dmrg_params,psi_last,prod_state=None,env_sweeps = 0):
    if prod_state is None:
        prod_state = ["up","up","down","down"] * (model_params['L'])
    
    M = SOCHubbardDiamondChain(model_params)   

    if psi_last is None:         
        psi = MPS.from_product_state(M.lat.mps_sites(), prod_state, bc=M.lat.bc_MPS)   
    else:
        psi = psi_last
        
    eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)

    start_time = time()
    if env_sweeps > 0:                              
        eng.environment_sweeps(env_sweeps)
    E, psi = eng.run()                              
    return MPS.correlation_length(psi),(time()-start_time)/60,psi

def CorrelationLine_R(Tarray,model_params,dmrg_params,psi_last,prod_state=None,env_sweeps = 0):
    model_params_list,CL_list,time_list,psi_list = [],[],[],[]
    psi_anzar = psi_last 

    for T in Tarray:
        model_params_temp = copy.deepcopy(model_params)
        model_params_temp.update({'t3': T })
        model_params_list.append(model_params_temp)
        
    for mp in model_params_list:
        CL, time,psi_anzar = CorrelationPoint_R(mp,dmrg_params,psi_anzar,prod_state)
        time_list.append(time)
        CL_list.append(CL)
        if psi_list == []:
            psi_list.append(psi_anzar)
        #print("Correlation length = ",CL)
        print("Time = ",time,flush=True)

    return CL_list, time_list,psi_list

def CorrelationGrild_R(T_params,model_params,dmrg_params,prod_state=None,env_sweeps = 0):
    t2_array = np.arange(T_params['t2_min'],T_params['t2_max']+ T_params['t2_steps'],T_params['t2_steps'])
    t3_array = np.arange(T_params['t3_min'],T_params['t3_max']+ T_params['t3_steps'],T_params['t3_steps'])

    CL_Grid,time_Grid,psi_hist_firts = [],[],None

    print("model_params = ",model_params,flush=True)
    print('T_params = ',T_params,flush=True)
    print("t3_array = ",t3_array,flush=True)

    for T2 in t2_array:
        model_params.update({'t2': T2})
        model_params.update({'t2p': T2})
        CL, time,psi_list = CorrelationLine_R(t3_array,model_params,dmrg_params,psi_hist_firts,prod_state)
        CL_Grid.append(CL)
        time_Grid.append(time)
        psi_hist_firts = psi_list[0]
        print("Correlation length grid = ",CL_Grid[-1],flush=True)
        print('t2_height =',T2,flush=True)
    return CL_Grid,time_Grid

def CorrelationPoint_NR(model_params,dmrg_params,prod_state=None,env_sweeps = 0):
    if prod_state is None:
        prod_state = ["up","up","down","down"] * (model_params['L'])
    M = SOCHubbardDiamondChain(model_params)            
    psi = MPS.from_product_state(M.lat.mps_sites(), prod_state, bc=M.lat.bc_MPS)   
    eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)

    start_time = time()
    if env_sweeps > 0:                              
        eng.environment_sweeps(env_sweeps)
    E, psi = eng.run()                              

    return MPS.correlation_length(psi),(time()-start_time)/60

def CorrelationLine_NR(Tarray,model_params,dmrg_params,prod_state=None,env_sweeps = 0):
    model_params_list = []
    CL_list = []
    time_list = []
    for T in Tarray:
        model_params_temp = copy.deepcopy(model_params)
        model_params_temp.update({'t3': T })
        model_params_list.append(model_params_temp)
        
    for mp in model_params_list:
        CL, time = CorrelationPoint_NR(mp,dmrg_params,prod_state)
        CL_list.append(CL)
        time_list.append(time)
        print("Time = ",time,flush=True)
    return CL_list, time_list

def CorrelationGrildn_NR(T_params,model_params,dmrg_params,prod_state=None,env_sweeps = 0):
    t2_array = np.arange(T_params['t2_min'],T_params['t2_max']+ T_params['t2_steps'],T_params['t2_steps'])
    t3_array = np.arange(T_params['t3_min'],T_params['t3_max']+ T_params['t3_steps'],T_params['t3_steps'])
    CL_Grid = []
    time_Grid = []

    print("model_params = ",model_params,flush=True)
    print('T_params = ',T_params,flush=True)
    print("t3_array = ",t3_array,flush=True)

    for T2 in t2_array:
        model_params.update({'t2': T2})
        model_params.update({'t2p': T2})
        CL, time = CorrelationLine_NR(t3_array,model_params,dmrg_params,prod_state)
        CL_Grid.append(CL)
        time_Grid.append(time)
        print("Correlation length grid = ",CL_Grid[-1],flush=True)
        print('t2_height =',T2,flush=True)
    return CL_Grid,time_Grid


##############################################################################################
def SaveCorrelation(T_params,dir,CorrelationLenght,Times):
    l1,l2,U,t1,chi_max = T_params['t2_max'],T_params['t3_max'],model_params['U'],model_params['t1'],dmrg_params['chi_list']
    t2_array = np.arange(T_params['t2_min'],T_params['t2_max']+T_params['t2_steps'],T_params['t2_steps'])
    t3_array = np.arange(T_params['t3_min'],T_params['t3_max']+T_params['t3_steps'],T_params['t3_steps'])
    points = len(t2_array)*len(t3_array)

    if not os.path.exists(dir):
        os.makedirs(dir)
    if Recycle:
        name= f"Points_{points}_R_t1_{t1}_t2_{l1}_t3_{l2}_U_{U/t1}.h5"
    else: 
        name= f"Points_{points}_NR_t1_{t1}_t2_{l1}_t3_{l2}_U_{U/t1}.h5"

    root = os.path.join(dir, name)
    with h5py.File(root, 'w') as paper:
        paper.create_dataset('CorrelationLenght', data=CorrelationLenght)
        paper.create_dataset('Times', data=Times)
        paper.create_dataset('t2_array', data=t2_array)
        paper.create_dataset('t3_array', data=t3_array)
    print(f"Archivo HDF5 guardado en: {root}")
    return None

##############################################################################################
pstate = ["up","up","down","down"] * (model_params['L'])

if Recycle:
    CL_Grid,time_Grid = CorrelationGrild_R(T_params,model_params,dmrg_params,pstate)
else: 
    CL_Grid,time_Grid = CorrelationGrildn_NR(T_params,model_params,dmrg_params,pstate)

SaveCorrelation(T_params,carpeta,CL_Grid,time_Grid)
