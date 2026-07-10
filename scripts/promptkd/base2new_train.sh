#!/bin/bash

# Example:
# sh scripts/promptkd/base2new_train.sh dtd 1 0 0.2 0.25 4 0.5 32 0 8192

# custom config
DATA="/data2/workspace_hyw/promptkd/promptkd_data"
TRAINER=PromptKD

DATASET=$1 # 'imagenet' 'caltech101' 'dtd' 'eurosat' 'fgvc_aircraft' 'oxford_flowers' 'food101' 'oxford_pets' 'stanford_cars' 'sun397' 'ucf101'
SEED=$2
GPU_ID=${3:-0}
DVP_ALPHA=${4:-0.2}
PRIOR_GAMMA=${5:-0.5}
SNS_RANK=${6:-4}
SNS_RHO=${7:-0.5}
SNS_PCA_RANK=${8:-32}
SNS_SEMANTIC_RANK=${9:-0}
SNS_MAX_SAMPLES=${10:-8192}
KD_WEIGHT=${11:-}

CFG=vit_b16_c2_ep20_batch8_4+4ctx
SHOTS=0

DIR=output/base2new/train_base/${DATASET}/dvp${DVP_ALPHA}_prior${PRIOR_GAMMA}_snsr${SNS_RANK}_rho${SNS_RHO}_pca${SNS_PCA_RANK}_sem${SNS_SEMANTIC_RANK}/shots_${SHOTS}/${TRAINER}/${CFG}/seed_${SEED}

# fgvc_aircraft, oxford_flowers, dtd: KD_WEIGHT=200
# imagenet, caltech101, eurosat, food101, oxford_pets, stanford_cars, sun397, ucf101: KD_WEIGHT=1000
if [ -z "${KD_WEIGHT}" ]; then
    case "${DATASET}" in
        fgvc_aircraft|oxford_flowers|dtd)
            KD_WEIGHT=200.0
            ;;
        imagenet|caltech101|eurosat|food101|oxford_pets|stanford_cars|sun397|ucf101)
            KD_WEIGHT=1000.0
            ;;
        *)
            KD_WEIGHT=1000.0
            ;;
    esac
fi

CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAINER.MODAL base2novel \
    TRAINER.PROMPTKD.TEMPERATURE 1.0 \
    TRAINER.PROMPTKD.KD_WEIGHT ${KD_WEIGHT} \
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
    TRAINER.PROMPTKD.SNS_MAX_SAMPLES ${SNS_MAX_SAMPLES}
