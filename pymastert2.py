import numpy as np
import os 
from os import system as s 
import sys
import tenpy as tp
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

job_id = os.getenv("SLURM_JOB_ID")
cpus_per_task = os.getenv("SLURM_CPUS_PER_TASK")
num_tasks = os.getenv("SLURM_NTASKS")

if len(sys.argv) > 1:
    job_id = sys.argv[1]
    cpus_per_task = sys.argv[2]
    num_tasks = sys.argv[3] 
    date = sys.argv[4]


# Datos que controlan la simulacion 
namemaster = '/scratch/villodre/Hubbard_calculations/Codigos/codigos_calculos/Hubard_t2_paral.py'
job_id = sys.argv[1]
recycle = False
U = 0.0
t_max,t_min = 0.52,0.28
#t_max,t_min = 3.0,0.0
t_steps = 0.04
chi_list = 'P'

# Calculamos el numero de puntos y como se repartiran 
total_cpus = int(num_tasks)
points_tot = (t_max-t_min)/t_steps + 1  # Asi incluyo tambien el final
points_account = int(points_tot/total_cpus)*np.ones(total_cpus)
t_max_r,t_min_r = np.zeros(total_cpus),np.zeros(total_cpus)

# Si sobran puntos, van al final 
n = 1 
while np.sum(points_account) != points_tot:
    points_account[-n] += 1
    n += 1

# Asignamos los valore de maximos y minimos a cada cpu 

for j in range(total_cpus):
    if j == 0:
        t_min_r[j] = t_min
        t_max_r[j] = t_min + t_steps*(points_account[j]-1) 
    else:
        t_min_r[j] = t_max_r[j-1] + t_steps
        t_max_r[j] = t_min_r[j] + t_steps*(points_account[j]-1) 

points_account = points_account.astype(int)
print('Puntos a calcular por cpu:',points_account[rank])
print('Rangos de t:',t_min_r,t_max_r)

# Generamos los archivos que seran cargados : 
models = []

for i in range(len(t_max_r)):
    if recycle:
        models.append('{} {} {} {} {} {} {} {} {} {}'.format(namemaster,'R',U,t_max_r[i],t_steps,t_min_r[i],points_account[i],'P',job_id,date))
    else:
        models.append('{} {} {} {} {} {} {} {} {} {}'.format(namemaster,'NR',U,t_max_r[i],t_steps,t_min_r[i],'P',points_account[i],job_id,date))

if recycle:
    s(f'python {models[rank]} > /scratch/villodre/Hubbard_calculations/Outs/Outs_t2/{job_id}_{date}/dmrgR_{job_id}_{rank}.out 2> /scratch/villodre/Hubbard_calculations/Error/Error_t2/{job_id}_{date}/Hubbarddmrg_t2_{job_id}_{rank}.err')
else :
    s(f'python {models[rank]} > /scratch/villodre/Hubbard_calculations/Outs/Outs_t2/{job_id}_{date}/dmrgNR_{job_id}_{rank}.out 2> /scratch/villodre/Hubbard_calculations/Error/Error_t2/{job_id}_{date}/Hubbarddmrg_t2_{job_id}_{rank}.err')
