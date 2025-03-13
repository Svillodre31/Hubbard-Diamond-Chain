import numpy as np
from os import system as s 
import sys
import tenpy as tp
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


namemaster = '/scratch/villodre/Hubbard_calculations/Codigos/codigos_calculos/Hubbardmaster.py'

if len(sys.argv) > 1:
    job_id = sys.argv[1]
    cpus_per_task = sys.argv[2]
    num_tasks = sys.argv[3] 
    date = sys.argv[4]

recycle = True
models = []

U_list = [0.25,1.,2.,4.]
t_max = 3.0*np.ones(len(U_list))
t_steps = 0.05*np.ones(len(U_list))
t_min = 0.0*np.ones(len(U_list))
chi_list = 'P'

for i in range(len(U_list)):
    if recycle:
        models.append('{} {} {} {} {} {} {} {}'.format(namemaster,'R',U_list[i],t_max[i],t_steps[i],t_min[i],job_id,date))
    else:
        models.append('{} {} {} {} {} {} {} {}'.format(namemaster,'NR',U_list[i],t_max[i],t_steps[i],t_min[i],job_id,date))

if recycle:
    s(f'python {models[rank]} > /scratch/villodre/Hubbard_calculations/Outs/Outs_master/{job_id}_{date}/dmrgR_{job_id}_{rank}.out 2> /scratch/villodre/Hubbard_calculations/Error/Errores_master/{job_id}_{date}/Hubbarddmrg_U_{job_id}_{rank}.err')

else :
    s(f'python {models[rank]} > /scratch/villodre/Hubbard_calculations/Outs/Outs_master/{job_id}_{date}/dmrgNR_{job_id}_{rank}.out 2> /scratch/villodre/Hubbard_calculations/Error/Errores_master/{job_id}_{date}/Hubbarddmrg_U_{job_id}_{rank}.err')
