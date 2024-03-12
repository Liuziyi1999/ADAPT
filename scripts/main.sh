#!/bin/bash

cd ..

# custom config
DATA=/Code/ADAPT/dataset # you may change your path to dataset here
TRAINER=ADAPT

DATASET=$1 # name of the dataset
CFG=$2  # config file 配置文件名称
T=$3 # temperature
TAU=$4 # pseudo label threshold
U=$5 # coefficient for loss_u
NAME=$6 # job name


for SEED in 2
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        #srun -J testit -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 \
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.ADAPT.T ${T} \
        TRAINER.ADAPT.TAU ${TAU} \
        TRAINER.ADAPT.U ${U} &
    fi
done

# Wait for all background jobs to complete
wait

#sed -i 's/\r//' main.sh
#bash main.sh visda17 vit_b16 1.0 0.6 1.0 deep12
#bash main.sh office_home vit_b16 1.0 0.8 1.0 t1_deep12
#bash main.sh office vit_b16 1.0 0.8 1.0 t3
