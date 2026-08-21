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

target_domains=('a' 'c' 'p' 'r')

# bash scripts/source/eval_source.sh officehome rn50_ep50_officehome Source RN50 a 0
for domain in "${target_domains[@]}"; do
    new_str="${DOMAINS}-${domain}"
    python train.py \
    --gpu ${GPU} \
    --backbone ${BACKBONE} \
    --domains ${new_str} \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${DIR} \
    --eval-only
done



