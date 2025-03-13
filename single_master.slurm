#!/bin/bash
#SBATCH --partition=general
##SBATCH --qos=test
#SBATCH --qos=long
#SBATCH --job-name=Hubbarddmrg_paral
#SBATCH --mail-type=fail
#SBATCH --output /scratch/villodre/Hubbard_calculations/Outs/Outs_general/dmrg-%J.out  # %J=jobid.step, %N=node.
#SBATCH --chdir=./
#SBATCH --cpus-per-task=1
#SBATCH --mem=18G
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --error=/scratch/villodre/Hubbard_calculations/Error/Error_general/%x-%j.err   
##SBATCH --time=4-12::00
#SBATCH --nodes=1
## Crear la carpeta, si no me da error 

module purge
module load iimpi/2021a
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
sleep 5s
conda activate /scratch/villodre/conda-env/tenpy

echo "Started at $(date)"

# Obtener el JOB ID
JOB_ID=${SLURM_JOB_ID}
FECHA=$(date +"d_%d-m_%m-h_%H")

echo "Ejecutando con fecha: $FECHA"

# Crear carpetas si no existen
mkdir -p /scratch/villodre/Hubbard_calculations/Outs/Outs_master/${JOB_ID}_${FECHA}
mkdir -p /scratch/villodre/Hubbard_calculations/Error/Errores_master/${JOB_ID}_${FECHA}
mkdir -p /scratch/villodre/Hubbard_calculations/resultados/resultados_correlacion/${JOB_ID}_${FECHA}

echo "Carpetas creadas para el Job ID: $JOB_ID"

sleep 5s

mpiexec -n 20 python /scratch/villodre/Hubbard_calculations/Codigos/codigos_control/pymaster.py ${SLURM_JOB_ID} ${SLURM_CPUS_PER_TASK} ${SLURM_NTASKS} ${FECHA}
echo "finished at $(date)"
