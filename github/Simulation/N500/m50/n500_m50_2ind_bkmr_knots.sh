#!/bin/bash -l

# Specify which project
#$ -P mricog

# Give this job a name
#$ -N n500_m50_2ind_bkmr_knots

# Join standard output and error to a single file
#$ -j y
#$ -l avx
# -pe omp 4

# Name the file where to redirect standard output and error
#$ -o n500_m50_2ind_bkmr_knots.qlog

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

module load R/3.6.0
cd /Simulation/N500/m50
Rscript N500_M50_2ind_knots.R  


echo "=========================================================="
echo "Finished on : $(date)"
echo "=========================================================="

