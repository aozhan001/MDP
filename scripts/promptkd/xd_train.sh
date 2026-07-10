#!/bin/bash

# Example:
# sh scripts/promptkd/xd_train.sh dtd 1 0 0.2 0.25 4 0.5 32 0 8192

# custom config
DATA="/data2/workspace_hyw/promptkd/promptkd_data"
TRAINER=PromptKD

DATASET=$1 # 'dtd' 'eurosat' 'fgvc_aircraft' 'oxford_flowers' 'food101' 'oxford_pets' 'stanford_cars' 'sun397' 'ucf101' 'caltech101'
SEED=$2
GPU_ID=${3:-0}
DVP_ALPHA=${4:-0.2}
PRIOR_GAMMA=${5:-0.5}
SNS_RANK=${6:-4}
SNS_RHO=${7:-0.5}
SNS_PCA_RANK=${8:-32}
SNS_SEMANTIC_RANK=${9:-0}
SNS_MAX_SAMPLES=${10:-8192}

CFG=vit_b16_c2_ep20_batch8_4+4ctx_cross_datasets
SHOTS=0

DIR=output/xd/${DATASET}/dvp${DVP_ALPHA}_prior${PRIOR_GAMMA}_snsr${SNS_RANK}_rho${SNS_RHO}_pca${SNS_PCA_RANK}_sem${SNS_SEMANTIC_RANK}/${TRAINER}/${CFG}_${SHOTS}shots/seed_${SEED}

CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES all \
    TRAINER.PROMPTKD.TEMPERATURE 1.0 \
    TRAINER.PROMPTKD.KD_WEIGHT 1000.0 \
    TRAINER.PROMPTKD.USE_MULTI_TEMPLATE_TEXT False \
    TRAINER.PROMPTKD.DVP_ENABLE True \
    TRAINER.PROMPTKD.PRIOR_CORRECT True \
    TRAINER.PROMPTKD.SNS_ENABLE True \
    TRAINER.PROMPTKD.DVP_ALPHA ${DVP_ALPHA} \
    TRAINER.PROMPTKD.PRIOR_GAMMA ${PRIOR_GAMMA} \
    TRAINER.PROMPTKD.SNS_RANK ${SNS_RANK} \
    TRAINER.PROMPTKD.SNS_RHO ${SNS_RHO} \
    TRAINER.PROMPTKD.SNS_PCA_RANK ${SNS_PCA_RANK} \
    TRAINER.PROMPTKD.SNS_SEMANTIC_RANK ${SNS_SEMANTIC_RANK} \
    TRAINER.PROMPTKD.SNS_MAX_SAMPLES ${SNS_MAX_SAMPLES} \
    TRAINER.MODAL cross
