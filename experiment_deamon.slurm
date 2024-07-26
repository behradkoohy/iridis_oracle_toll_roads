#!/bin/bash

#SBATCH --time=11:00:00
#SBATCH --nodes=1
#SBATCH --partition=batch

source venv/bin/activate

python main.py $1 $2 $3 $4 $5 $6
