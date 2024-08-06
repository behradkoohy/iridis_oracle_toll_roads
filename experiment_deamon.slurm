#!/bin/bash

#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --output=run_outputs_200cars/%j.out

module load conda
#source venv/bin/activate
source activate benchmarks

python main.py $1 $2 $3 $4 $5 $6
