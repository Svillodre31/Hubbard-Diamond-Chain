import sys
import os
import numpy as np
from time import time
import matplotlib.pyplot as plt
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinSite
from tenpy.networks.site import Site
from tenpy.networks.site import SpinHalfFermionSite
from tenpy.algorithms import tebd
from tenpy.algorithms import dmrg
from tenpy.tools.hdf5_io import save,load
from tenpy.models.tf_ising import TFIChain
from tenpy.models.lattice import Lattice
from tenpy.models.lattice import Chain
from tenpy.models.lattice import Lattice,get_order
from tenpy.models.model import CouplingMPOModel
from tenpy.models.model import CouplingModel, \
NearestNeighborModel, MPOModel
sys.path.append(os.path.abspath("/scratch/villodre/conda-env/tenpy/"))
print(sys.path)
from modelhubbard import *


control_params = sys.argv[:]

model_params = { 
        'L': 2,                 # Length of the chain
        'bc_x':'periodic',      # Bound conditions of the lattice
        'bc_MPS':'infinite',    # Bound conditions of the MPS
        'mu':0,
        't1': 1.,               # Intracell nearest neighbor hopping (out of the diamond)
        't2': 1.5,              # Intracell next nearest neighbor coupling (inside the diamond)
        't2p': .3,              # Spin orbit coupling
        't3': 1.,               # Intercell coupling
        'U': 1.                 # Hubbard parameter
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
            20: 128
        },
        'max_E_err': 1.e-8,         # Error maximo en la entropia
        'max_S_err': 1.e-3,          # Error maximo en la energia
        'max_sweeps': 100,            # Numero maximo de sweeps que realiza antes de terminar
        'max_hours': 0.5        # Maximo tiempo que corre el algoritmo en horas (0.00014 horas = 30 segundos)
}

T_params = {'t2p_min': 0.0, 
            't2p_max': 3.0, 
            't2p_steps': 0.05}

if len(control_params) > 1: 
    if control_params[1] != 'P':     
        if int(control_params[1]) < 49:
            dmrg_params.update({'chi_list': {0: 9, 10 : int(control_params[1])}})
        else:
            dmrg_params.update({'chi_list': {0: 9, 10 : 49, 20 : int(control_params[1])}})
        print("chi_list = ",dmrg_params['chi_list'],flush=True)

if len(control_params) > 2 :
    if control_params[2] != 'P':
        job_id = control_params[2]
        print("job_id = ",job_id,flush=True)

if len(control_params) > 3 :
    if control_params[3] != 'P':
        date = control_params[3]
        print("date = ",date,flush=True)

print("Model Parameters = ",model_params,flush=True)
print("DMRG Parameters = ",dmrg_params,flush=True) 
print("T Parameters = ",T_params,flush=True)

carpeta = f"/scratch/villodre/Hubbard_calculations/resultados/resultados_chivariation_t2/{job_id}_{date}"


def CorrelationPoint(model_params,dmrg_params,prod_state=None,env_sweeps = 0):
    if prod_state is None:
        prod_state = ["up","up","down","down"] * (model_params['L'])

    M = SOCHubbardDiamondChain(model_params)            
    psi = MPS.from_product_state(M.lat.mps_sites(), prod_state, bc=M.lat.bc_MPS)   
    eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)

    start_time = time()
    if env_sweeps > 0:                              
        eng.environment_sweeps(env_sweeps)
    E, psi = eng.run()        

    print('Chi = ', psi.chi)
    return MPS.correlation_length(psi),psi.entanglement_entropy(),(time()-start_time)/60

def CorrelationLine(Tarray,model_params,dmrg_params,prod_state=None,env_sweeps = 0):
    model_params_list = []
    CL_list,EE_list,time_list = [],[],[]

    for T in Tarray:
        model_params_temp = copy.deepcopy(model_params)
        model_params_temp.update({'t2p': T })
        model_params_list.append(model_params_temp)
        
    for mp in model_params_list:
        CL,EE, time = CorrelationPoint(mp,dmrg_params,prod_state)
        CL_list.append(CL)
        time_list.append(time)
        EE_list.append(EE)
        print("Time = ",time,flush=True)
        
    return CL_list,EE_list, time_list


def Savelines(T_params,dir,CorrelationLenght,Entropy,Times):
    l1,l2,U,t1,t2,chi = T_params['t2p_max'],T_params['t2p_min'],model_params['U'],model_params['t1'],model_params['t2'],list(dmrg_params['chi_list'].values())[-1]
    print(chi)
    t3_array = np.arange(T_params['t2p_min'],T_params['t2p_max']+T_params['t2p_steps'],T_params['t2p_steps'])
    points = len(t3_array)
    name= f"Points_{points}_t2pmin_{l1}_t2pmax_{l2}_t2_{l2}_U_{U/t1}_Chi_{chi}.h5"
    
    root = os.path.join(dir, name)
    with h5py.File(root, 'w') as paper:
        paper.create_dataset('CorrelationLenght', data=CorrelationLenght)
        paper.create_dataset('Entropy', data=Entropy)
        paper.create_dataset('Times', data=Times)
        paper.create_dataset('t_2', data=t2)
        paper.create_dataset('chi', data=chi)
        paper.create_dataset('t3_array', data=t3_array)
    print(f"Archivo HDF5 guardado en: {root}")
    return None

t3_array = np.arange(T_params['t2p_min'],T_params['t2p_max']+T_params['t2p_steps'],T_params['t2p_steps'])
print("t2p_array = ",t3_array)

CL,EE,time_line = CorrelationLine(t3_array,model_params,dmrg_params)
Savelines(T_params,carpeta,CL,EE,time_line)

print("Proceso terminado",flush=True)






