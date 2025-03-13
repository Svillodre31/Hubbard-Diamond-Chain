import numpy as np
from os import system as s 
import sys
import tenpy as tp
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


namemaster = '/scratch/villodre/Hubbard_calculations/Codigos/codigos_calculos/Hubbardpadawan_t2.py'

if len(sys.argv) > 1:
    job_id = sys.argv[1]
    cpus_per_task = sys.argv[2]
    num_tasks = sys.argv[3] 
    date = sys.argv[4]

models = []
chi_list = [32,48,64,128,256,380,512,640,768]


for i in range(len(chi_list)):
    models.append('{} {} {} {} '.format(namemaster,chi_list[i],job_id,date))

s(f'python {models[rank]} > /scratch/villodre/Hubbard_calculations/Outs/Outs_chi_t2/{job_id}_{date}/dmrgR_{job_id}_{rank}.out 2> /scratch/villodre/Hubbard_calculations/Error/Error_chi_t2/{job_id}_{date}/Hubbarddmrg_paral_{job_id}_{rank}.err')