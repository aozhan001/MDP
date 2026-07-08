#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${REPO_ROOT}/../promptkd_data"

if ! command -v hf >/dev/null 2>&1; then
    echo "hf command not found. Install huggingface_hub CLI first." >&2
    exit 1
fi

mkdir -p "${DATA_DIR}"

download_and_extract() {
    local zip_name="$1"
    local dataset_dir="$2"
    local zip_path="${DATA_DIR}/${zip_name}"

    if [ -d "${DATA_DIR}/${dataset_dir}" ]; then
        echo "[skip] ${dataset_dir} already exists"
        return 0
    fi

    if [ -f "${zip_path}" ]; then
        if unzip -tq "${zip_path}" >/dev/null 2>&1; then
            echo "[keep] existing valid zip ${zip_name}"
        else
            echo "[drop] invalid zip ${zip_name}"
            rm -f "${zip_path}"
        fi
    fi

    if [ ! -f "${zip_path}" ]; then
        echo "[download] ${zip_name}"
        hf download zhengli97/prompt_learning_dataset \
            --repo-type model \
            --include "${zip_name}" \
            --local-dir "${DATA_DIR}"
    fi

    echo "[extract] ${zip_name}"
    unzip -o "${zip_path}" -d "${DATA_DIR}" >/dev/null
}

download_and_extract "caltech-101.zip" "caltech-101"
download_and_extract "dtd.zip" "dtd"
download_and_extract "eurosat.zip" "eurosat"
download_and_extract "fgvc_aircraft.zip" "fgvc_aircraft"
download_and_extract "food101.zip" "food-101"
download_and_extract "oxford_flowers.zip" "oxford_flowers"
download_and_extract "oxford_pets.zip" "oxford_pets"
download_and_extract "stanford_cars.zip" "stanford_cars"
download_and_extract "sun397.zip" "sun397"
download_and_extract "ucf101.zip" "ucf101"
download_and_extract "imagenet-adversarial.zip" "imagenet-adversarial"
download_and_extract "imagenet-rendition.zip" "imagenet-rendition"
download_and_extract "imagenet-sketch.zip" "imagenet-sketch"
download_and_extract "imagenetv2.zip" "imagenetv2"

echo
echo "Done. Current dataset directory:"
find "${DATA_DIR}" -maxdepth 1 -mindepth 1 -type d | sort
echo
echo "Note: full ImageNet is not included in zhengli97/prompt_learning_dataset and must be prepared separately."
