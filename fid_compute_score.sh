cp "../inception-2015-12-05.pt" "/tmp"

DATASET_PATH="./benchmark/fid/dataset/COCO/"
INFERENCE_NUM=20
WARMUP_NUM=4
SAMPLE_NUM=2000
SAMPLE_IMAGES_FOLDER="${DATASET_PATH}Sample${SAMPLE_NUM}/TotalStep${INFERENCE_NUM}/WarmupStep${WARMUP_NUM}/"
REF_DIR="${SAMPLE_IMAGES_FOLDER}original"


TOKEN_MASK_OPTIONS=("Height" "Random" "EvenCluster" "UnevenCluster")
KV_MASK_OPTIONS=("Fixed" "Random" "MaxCosine" "MinCosine")
KV_MAX_OPTIONS=(64 56 48 40 32)
# TOKEN_MASK_OPTIONS=("Height")
# KV_MASK_OPTIONS=("Random")
# KV_MAX_OPTIONS=(64)
for token_mask in "${TOKEN_MASK_OPTIONS[@]}"; do
    for kv_mask in "${KV_MASK_OPTIONS[@]}"; do
        for kv_max in "${KV_MAX_OPTIONS[@]}"; do
            SAMPLE_DIR="${SAMPLE_IMAGES_FOLDER}KV${kv_max}/${kv_mask}KVMask/${token_mask}TokenMask/"
            python ./benchmark/fid/compute_fid.py \
                --ref $REF_DIR \
                --sample $SAMPLE_DIR \
            2>/dev/null
        done
    done
done
