#!/bin/bash

#SBATCH --time=11:00:00
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --output=run_outputs/%j.out

module load conda
source activate benchmarks

source venv/bin/activate

python main.py $1 $2 $3 $4 $5 $6
