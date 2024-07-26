#!/bin/bash

dbpath="big_results.db"

source venv/bin/activate

python CreateDatabase.py $dbpath

#model_name="PSOTimeOnly"
#description="none"
#vot_seed=0
#for i in {1..100}
#do
#  sbatch experiment_deamon.slurm $dbpath $i $model_name $description $vot_seed $i
#done
models=(
    "PSOTimeOnly"
    "PSOFixedPriceTTOpt"
    "PSOFixedPriceSCOpt"
    "PSOFixedPriceCCOpt"
    "PSOTimestepPriceTTOpt"
    "PSOTimestepPriceSCOpt"
    "PSOTimestepPriceCCOpt"
    "PSOFUnboundPriceTTOpt"
    "PSOFUnboundPriceSCOpt"
    "PSOFUnboundPriceCCOpt"
    "PSOLinearPriceTTOpt"
    "PSOLinearPriceSCOpt"
    "PSOLinearPriceCCOpt"
)

description="none"
vot_seed=0
id_counter=0

for model_name in "${models[@]}"
do
  for ((i=0; i<50; i++))
  do
    sbatch experiment_deamon.slurm $dbpath $id_counter $model_name $description $vot_seed $i
    id_counter=$((id_counter + 1))
    
  done
done
