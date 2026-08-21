#!/bin/bash

# custom config
DATA=/data/dxw/data # your directory

DATASET=$1
CFG=$2  # config file
TRAINER=$3
BACKBONE=$4 # backbone name
DOMAINS=$5
GPU=$6


DIR=output/source/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/${DOMAINS}

#if [ -d "$DIR" ]; then
#    echo "Results are available in ${DIR}, so skip this job"
#else
#    echo "Run this job and save the output to ${DIR}"

    python train.py \
        --gpu ${GPU} \
        --backbone ${BACKBONE} \
        --domains ${DOMAINS} \
        --root ${DATA} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR}
#fi
