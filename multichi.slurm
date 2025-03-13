#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular 
##SBATCH --qos=test
#SBATCH --job-name=Hubbarddmrg_chi
#SBATCH --mail-type=fail
#SBATCH --output /scratch/villodre/Hubbard_calculations/Outs/Outs_general/dmrg-%J.out  # %J=jobid.step, %N=node.
#SBATCH --chdir=./
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --error=/scratch/villodre/Hubbard_calculations/Error/Error_general/%x-%j.err   
##SBATCH --time=23:59:00
#SBATCH --nodes=1


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
mkdir -p /scratch/villodre/Hubbard_calculations/Outs/Outs_chi/${JOB_ID}_${FECHA}
mkdir -p /scratch/villodre/Hubbard_calculations/Error/Error_chi/${JOB_ID}_${FECHA}
mkdir -p /scratch/villodre/Hubbard_calculations/resultados/resultados_chivariation/${JOB_ID}_${FECHA}

echo "Carpetas creadas para el Job ID: $JOB_ID"
sleep 5s

mpiexec -n 3 python /scratch/villodre/Hubbard_calculations/Codigos/codigos_control/pymarterchi.py ${SLURM_JOB_ID} ${SLURM_CPUS_PER_TASK} ${SLURM_NTASKS} ${FECHA}

echo "finished at $(date)"
