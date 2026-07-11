#!/bin/bash

# Default experiment: PromptKD + DVP + Prior Correction + FNS.
# Usage example:
# sh scripts/promptkd/base2new_train.sh \
#   dtd 1 0 \
#   /path/to/original_promptkd/dtd/seed_1 \
#   0.2 0.25 16 1.0 32 prompt_only

DATA="/data2/workspace_hyw/promptkd/promptkd_data"
TRAINER=PromptKD

DATASET=$1
SEED=$2
GPU_ID=${3:-0}
FNS_REFERENCE_PATH=$4
DVP_ALPHA=${5:-0.2}
PRIOR_GAMMA=${6:-0.5}
FNS_RANK=${7:-16}
FNS_RHO=${8:-1.0}
FNS_NUM_BATCHES=${9:-32}
FNS_PARAM_SCOPE=${10:-prompt_only}
KD_WEIGHT=${11:-}

if [ -z "${FNS_REFERENCE_PATH}" ]; then
    echo "Error: the existing baseline PromptKD checkpoint path is required."
    exit 1
fi

CFG=vit_b16_c2_ep20_batch8_4+4ctx
SHOTS=0

DIR="output/base2new/train_base/${DATASET}/dvp${DVP_ALPHA}_prior${PRIOR_GAMMA}_fnsr${FNS_RANK}_rho${FNS_RHO}_fb${FNS_NUM_BATCHES}_${FNS_PARAM_SCOPE}/shots_${SHOTS}/${TRAINER}/${CFG}/seed_${SEED}"

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

CUDA_VISIBLE_DEVICES="${GPU_ID}" python train.py \
    --root "${DATA}" \
    --seed "${SEED}" \
    --trainer "${TRAINER}" \
    --dataset-config-file "configs/datasets/${DATASET}.yaml" \
    --config-file "configs/trainers/${TRAINER}/${CFG}.yaml" \
    --output-dir "${DIR}" \
    DATASET.NUM_SHOTS "${SHOTS}" \
    TRAINER.MODAL base2novel \
    TRAINER.PROMPTKD.TEMPERATURE 1.0 \
    TRAINER.PROMPTKD.KD_WEIGHT "${KD_WEIGHT}" \
    TRAINER.PROMPTKD.USE_MULTI_TEMPLATE_TEXT False \
    TRAINER.PROMPTKD.SNS_ENABLE False \
    TRAINER.PROMPTKD.DVP_ENABLE True \
    TRAINER.PROMPTKD.PRIOR_CORRECT True \
    TRAINER.PROMPTKD.FNS_ENABLE True \
    TRAINER.PROMPTKD.FNS_REFERENCE_PATH "${FNS_REFERENCE_PATH}" \
    TRAINER.PROMPTKD.FNS_INIT_FROM_REFERENCE True \
    TRAINER.PROMPTKD.FNS_STRICT_REFERENCE True \
    TRAINER.PROMPTKD.FNS_PARAM_SCOPE "${FNS_PARAM_SCOPE}" \
    TRAINER.PROMPTKD.FNS_CLASS_SCOPE auto \
    TRAINER.PROMPTKD.FNS_RANK "${FNS_RANK}" \
    TRAINER.PROMPTKD.FNS_RHO "${FNS_RHO}" \
    TRAINER.PROMPTKD.FNS_NUM_BATCHES "${FNS_NUM_BATCHES}" \
    TRAINER.PROMPTKD.DVP_ALPHA "${DVP_ALPHA}" \
    TRAINER.PROMPTKD.PRIOR_GAMMA "${PRIOR_GAMMA}"
