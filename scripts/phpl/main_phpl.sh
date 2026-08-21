#!/bin/bash

# custom config
DATA=/data/dxw/data  # your directory

DATASET=$1
CFG=$2  # config file
TRAINER=$3
BACKBONE=$4 # backbone name
NTOK=$5
DOMAINS=$6
GPU=$7

LOCATION=middle
DEEP=False
DEEPLAYER=None

DIR=output/PHPL/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/deep${DEEP}_${LOCATION}_phpl/${DOMAINS}_ntok${NTOK}

python train.py \
    --gpu ${GPU} \
    --backbone ${BACKBONE} \
    --domains ${DOMAINS} \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    TRAINER.PHPL.NUM_TOKENS ${NTOK} \
    TRAINER.PHPL.N_CTX ${NTOK} \
    TRAINER.PHPL.T_DEEP ${DEEP} \
    TRAINER.PHPL.V_DEEP ${DEEP} \
    TRAINER.PHPL.LOCATION ${LOCATION} \
    TRAINER.PHPL.DEEP_LAYERS ${DEEPLAYER}


