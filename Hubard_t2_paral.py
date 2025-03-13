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
from modelhubbard import *

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
        'U': 0.4                # Hubbard parameter
}
dmrg_params = {
        'mixer': True,               # setting this to True helps to escape local minima
        'mixer_params': {
            'amplitude': 1.e-5,      # amplitud del mixer
            'decay': 1.2,            # parametro que controla como de rapido decae la amplitud del mixer 
            'disable_after': 15      # desactivo el mixer despues de 20 sweeps
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
            0: 49,
            20: 128
        },
        'max_E_err': 1.e-8,          # Error maximo en la entropia
        'max_S_err': 1.e-3,          # Error maximo en la energia
        'max_sweeps': 100,           # Numero maximo de sweeps que realiza antes de terminar
        'max_hours': 0.1             # Maximo tiempo que corre el algoritmo en horas (0.00014 horas = 30 segundos)
}

T_params = {'t2_min': 0.0, 
            't2_max': 3.0, 
            't2_steps': 0.5, 
            't2p_min': 0.0, 
            't2p_max': 3.0, 
            't2p_steps': 0.5}

T_params['t2_points'] = (T_params['t2_max'] - T_params['t2_min'] )/T_params['t2_steps'] + 1

if control_params[1] == 'R':
    Recycle = True
else:
    Recycle = False

if len(control_params) > 2: 
    if control_params[2] != 'P':     
        model_params.update({'U': float(control_params[2])})
        print("U = ",model_params['U'],flush=True)

if len(control_params) > 3:
    if control_params[3] != 'P':
        T_params.update({'t2_max': float(control_params[3])})
        print("T_max = ",T_params['t2_max'],flush=True)

if len(control_params) > 4: 
    if control_params[4] != 'P':
        T_params.update({'t2_steps': float(control_params[4])})
        T_params.update({'t2p_steps': float(control_params[4])})        
        print("t2_steps = ",T_params['t2_steps'],flush=True)    

if len(control_params) > 5 : 
    if control_params[5] != 'P':
        T_params.update({'t2_min': float(control_params[5])})
        print("T_min = ",T_params['t2_min'],flush=True)

if len(control_params) > 6 : 
    if  control_params[6] != 'P':
        if int(control_params[6]) < 49:
            dmrg_params.update({'chi_list': {0: 9, 10 : int(control_params[6])}})
        else:
            dmrg_params.update({'chi_list': {0: 9, 10 : 49, 20 : int(control_params[6])}})
        print("chi_list = ",dmrg_params['chi_list'],flush=True)
        
if len(control_params) > 7 :
    if control_params[7] != 'P':
        t2_points = control_params[7]
        T_params.update({'t2_points': int(control_params[7])})
        print("t2_points = ",T_params['t2_points'],flush=True)

if len(control_params) > 8 :
    if control_params[8] != 'P':
        job_id = control_params[8]
        print("job_id = ",job_id,flush=True)

if len(control_params) > 9 :
    if control_params[9] != 'P':
        date = control_params[9]
        print("date = ",date,flush=True)


carpeta = f"/scratch/villodre/Hubbard_calculations/resultados/resultados_t2/{job_id}_{date}"

def CorrelationPoint_R(model_params,dmrg_params,psi_last,prod_state=None,env_sweeps = 0):
    if prod_state is None:
        prod_state = ["up","up","down","down"] * (model_params['L'])

    trunc_par = {"chi_max": 5, "svd_min": 1e-6}
    
    M = SOCHubbardDiamondChain(model_params)   
    start_time_sweep = time()
    start_time = time()

    if psi_last is None:         
        psi = MPS.from_product_state(M.lat.mps_sites(), prod_state, bc=M.lat.bc_MPS)   
    else:
        psi = psi_last.copy()  
        psi.compress_svd(trunc_par)

    eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)

    if env_sweeps > 0:                              
        eng.environment_sweeps(env_sweeps)

    print("Time to create the MPS = ",(time()-start_time_sweep)/60,flush=True)
    E, psi = eng.run()      
    R = eng.get_resume_data()
    print('Numbers of sweeps:',R['sweeps'],flush=True)  

    return MPS.correlation_length(psi),(time()-start_time)/60,psi

def CorrelationLine_R(Tarray,model_params,dmrg_params,psi_last,prod_state=None,env_sweeps = 0):
    model_params_list,CL_list,time_list,psi_list = [],[],[],[]
    psi_anzar = psi_last 

    for T in Tarray:
        model_params_temp = copy.deepcopy(model_params)
        model_params_temp.update({'t2p': T})
        model_params_list.append(model_params_temp)
        
    for mp in model_params_list:
        CL, time,psi_anzar = CorrelationPoint_R(mp,dmrg_params,psi_anzar,prod_state,env_sweeps)
        time_list.append(time)
        CL_list.append(CL)
        if psi_list == []:
            psi_list.append(psi_anzar)
        
        print('Chi = ', psi_anzar.chi)
        print("Time = ",time,flush=True)

    return CL_list, time_list,psi_list

def CorrelationGrild_R(T_params,model_params,dmrg_params,prod_state=None,env_sweeps = 0):
    t2_array = np.linspace(T_params['t2_min'],T_params['t2_max'],T_params['t2_points'])
    t2p_array = np.arange(T_params['t2p_min'],T_params['t2p_max']+ T_params['t2p_steps'],T_params['t2p_steps'])
    print("t2_array = ",t2_array,flush=True)

    CL_Grid,time_Grid,psi_hist_firts = [],[],None

    print("model_params = ",model_params,flush=True)
    print('T_params = ',T_params,flush=True)
    print("t2p_array = ",t2p_array,flush=True)

    for T2 in t2_array:
        model_params.update({'t2': T2})
        CL, time,psi_list = CorrelationLine_R(t2p_array,model_params,dmrg_params,psi_hist_firts,prod_state,env_sweeps)
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
    R = eng.get_resume_data()
    
    print('Numbers of sweeps:',R['sweeps'],flush=True)  
    print('Chi = ', psi.chi)
    return MPS.correlation_length(psi),(time()-start_time)/60

def CorrelationLine_NR(Tarray,model_params,dmrg_params,prod_state=None,env_sweeps = 0):
    model_params_list = []
    CL_list = []
    time_list = []
    for T in Tarray:
        model_params_temp = copy.deepcopy(model_params)
        model_params_temp.update({'t2p': T })
        model_params_list.append(model_params_temp)
        
    for mp in model_params_list:
        CL, time = CorrelationPoint_NR(mp,dmrg_params,prod_state)
        CL_list.append(CL)
        time_list.append(time)
        print("Time = ",time,flush=True)
    return CL_list, time_list

def CorrelationGrildn_NR(T_params,model_params,dmrg_params,prod_state=None,env_sweeps = 0):
    t2_array = np.linspace(T_params['t2_min'],T_params['t2_max'],T_params['t2_points'])
    t2p_array = np.arange(T_params['t2p_min'],T_params['t2p_max']+ T_params['t2p_steps'],T_params['t2p_steps'])
    print("t2_array = ",t2_array,flush=True)
    CL_Grid = []
    time_Grid = []

    print("model_params = ",model_params,flush=True)
    print('T_params = ',T_params,flush=True)
    print("t2p_array = ",t2p_array,flush=True)

    for T2 in t2_array:
        model_params.update({'t2': T2})
        CL, time = CorrelationLine_NR(t2p_array,model_params,dmrg_params,prod_state)
        CL_Grid.append(CL)
        time_Grid.append(time)
        print("Correlation length grid = ",CL_Grid[-1],flush=True)
        print('t2_height =',T2,flush=True)
    return CL_Grid,time_Grid

##############################################################################################
def SaveCorrelation(T_params,dir,CorrelationLenght,Times):
    l1,l2,U,t1,chi_max,L_sis = T_params['t2_max'],T_params['t2p_max'],model_params['U'],model_params['t1'],dmrg_params['chi_list'],model_params['L']
    t2_array = np.linspace(T_params['t2_min'],T_params['t2_max'],T_params['t2_points'])
    t2p_array = np.arange(T_params['t2p_min'],T_params['t2p_max']+ T_params['t2p_steps'],T_params['t2p_steps'])
    points = len(t2_array)*len(t2p_array)

    if not os.path.exists(dir):
        os.makedirs(dir)
    if Recycle:
        name= f"Points_{points}_R_L_{L_sis}_t1_{t1}_t2_{l1}_t2p_{l2}_U_{U/t1}.h5"
    else: 
        name= f"Points_{points}_NR_L_{L_sis}_t1_{t1}_t2_{l1}_t2p_{l2}_U_{U/t1}.h5"

    root = os.path.join(dir, name)
    with h5py.File(root, 'w') as paper:
        paper.create_dataset('CorrelationLenght', data=CorrelationLenght)
        paper.create_dataset('Times', data=Times)
        paper.create_dataset('t2_array', data=t2_array)
        paper.create_dataset('t2p_array', data=t2p_array)
        paper.create_dataset('t3', data=model_params['t3'])
        paper.create_dataset('U', data=U)
        paper.create_dataset('t1', data=t1)
        paper.create_dataset('chi_max', data=chi_max)
    print(f"Archivo HDF5 guardado en: {root}")
    return None

##############################################################################################
pstate = ["up","up","down","down"] * (model_params['L'])
sweeps_env = 20 

if Recycle:
    CL_Grid,time_Grid = CorrelationGrild_R(T_params,model_params,dmrg_params,pstate,sweeps_env)
else: 
    CL_Grid,time_Grid = CorrelationGrildn_NR(T_params,model_params,dmrg_params,pstate)

SaveCorrelation(T_params,carpeta,CL_Grid,time_Grid)


