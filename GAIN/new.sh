#!/bin/bash
#SBATCH --job-name=newjob
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24000
#SBATCH --gres=gpu:2
#SBATCH -o /cluster/me66vepi/GAIN_working/GAIN/outputs/output.out
#SBATCH -e /cluster/me66vepi/GAIN_working/GAIN/errors/error.err
#SBATCH --mail-type=ALL
#Timelimit format: "23:59:00" -- max is 24h
#SBATCH --time=24:00:00
#SBATCH --exclude=lme53,lme49,lme51,lme52,lme50
# Tell's pipenv to install the virtualenvs in the cluster folder
export WORKON_HOME==/cluster/`whoami`/.python_cache
echo "Your job is running on" $(hostname)
scp -r /cluster/me66vepi/data /scratch/$SLURM_JOB_ID/data
# Small Python packages can be installed in own home directory. Not recommended for big packages like tensorflow -> Follow instructions for pipenv below
# cluster_requirements.txt is a text file listing the required pip packages (one package per line)
echo "Data copied succesfully"
pip3 install --user -r requirements.txt
python3 /cluster/me66vepi/GAIN_working/GAIN/main.py