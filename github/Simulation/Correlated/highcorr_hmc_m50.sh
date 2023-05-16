#!/bin/bash -l

# Specify which project
#$ -P mricog

# Give this job a name
#$ -N highcorr_hmc_m50

# Join standard output and error to a single file
#$ -j y
#$ -l avx
#$ -pe omp 4

# Name the file where to redirect standard output and error
#$ -o highcorr_hmc_m50.qlog

# Send an email when job ends running or has problem
# -m eas

# Whom to send the email to
# -M rw94@bu.edu

#Request 100 hours for the job
# -l h_rt=47:50:00

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
python /Simulation/Correlated/highcorr_hmc_m50.py



echo "=========================================================="
echo "Finished on : $(date)"
echo "=========================================================="

