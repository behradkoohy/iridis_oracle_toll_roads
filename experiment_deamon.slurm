#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=bk2g18@soton.ac.uk
#SBATCH --time=22:00:00
#SBATCH --nodes=1
#SBATCH --partition=amd

source bin/activate

python main.py $1 $2 $3 $4 $5 $6
