#!/bin/bash

# custom config
DATA="/data2/workspace_hyw/promptkd/promptkd_data"
TRAINER=PromptKD

DATASET=$1 # 'imagenet' 'caltech101' 'dtd' 'eurosat' 'fgvc_aircraft' 'oxford_flowers' 'food101' 'oxford_pets' 'stanford_cars' 'sun397' 'ucf101'
SEED=$2
GPU_ID=$3
MTP_ALPHA=$4
DVP_ALPHA=$5
PRIOR_GAMMA=$6
KD_WEIGHT=${7:-}
MTP_ENABLE=${8:-True}
DVP_ENABLE=${9:-True}
PRIOR_ENABLE=${10:-True}

#dtd:0.3 0.2 0.25
#oxford_flowers:0.2 0.2 0.25
#eurosat:0.2 0.2 1


#vit_b16_c2_ep20_batch8_4+4ctx_mgkd
CFG=vit_b16_c2_ep20_batch8_4+4ctx
SHOTS=0

DIR=output/base2new/train_base/${DATASET}/${DATASET}_${MTP_ALPHA}_${DVP_ALPHA}_${PRIOR_GAMMA}/shots_${SHOTS}/${TRAINER}/${CFG}/seed_${SEED}

# fgvc_aircraft, oxford_flowers, dtd: KD_WEIGHT:200
# imagenet, caltech101, eurosat, food101, oxford_pets, stanford_cars, sun397, ucf101, KD_WEIGHT:1000

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
    TRAINER.PROMPTKD.USE_MULTI_TEMPLATE_TEXT ${MTP_ENABLE} \
    TRAINER.PROMPTKD.DVP_ENABLE ${DVP_ENABLE} \
    TRAINER.PROMPTKD.PRIOR_CORRECT ${PRIOR_ENABLE}\
    TRAINER.PROMPTKD.MTP_ALPHA ${MTP_ALPHA}\
    TRAINER.PROMPTKD.DVP_ALPHA ${DVP_ALPHA}\
    TRAINER.PROMPTKD.PRIOR_GAMMA ${PRIOR_GAMMA}

    
