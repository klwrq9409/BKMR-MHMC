#!/bin/bash -l

# Specify which project
#$ -P mricog

# Give this job a name
#$ -N n500_m200_6ind

# Join standard output and error to a single file
#$ -j y
#$ -l avx
#$ -pe omp 4

# Name the file where to redirect standard output and error
#$ -o n500_m200_6ind_regular.qlog

# Send an email when job ends running or has problem
# -m eas

# Whom to send the email to
# -M rw94@bu.edu

#Request 100 hours for the job
# -l h_rt=11:50:00

# Submit an array job
#$ -t 1-100


# Now let's keep track of some information just in case anything goes wrong

echo "=========================================================="
echo "Starting on       : $(date)"
echo "Running on node   : $(hostname)"
echo "Current directory : $(pwd)"
echo "Current job ID    : $JOB_ID"
echo "Current job name  : $JOB_NAME"
echo "Task index number : $TASK_ID"
echo "=========================================================="

module load miniconda
conda activate bkmr_mhmc
python /Simulation/6ind/N500/m200/n500_m200_6ind.py



echo "=========================================================="
echo "Finished on : $(date)"
echo "=========================================================="

