export NUM_SAMPLE="--num_samples 2000"
DATASET_PATH="./benchmark/fid/dataset/COCO/"
export CAPTION_FILE="--caption_file ${DATASET_PATH}dataset_coco.json"
export TASK_ARGS="--height 256 --width 256 --no_use_resolution_binning"
export CUDA_VISIBLE_DEVICES=0,1,2,3

export SCRIPT="sd3_generate.py"
export MODEL_ID="/archive/share/cql/model/stable-diffusion-3-medium-diffusers"
# export SCRIPT="flux_generate.py"
# export MODEL_ID="/archive/share/cql/model/FLUX.1-dev"

export INFERENCE_STEP="--num_inference_steps 20"
export WARMUP_STEP="--warmup_step 4"
INFERENCE_NUM=$(echo $INFERENCE_STEP | grep -o '[0-9]\+')
WARMUP_NUM=$(echo $WARMUP_STEP | grep -o '[0-9]\+')
SAMPLE_NUM=$(echo $NUM_SAMPLE | grep -o '[0-9]\+')
export SAMPLE_IMAGES_FOLDER="--sample_images_folder ${DATASET_PATH}Sample${SAMPLE_NUM}/TotalStep${INFERENCE_NUM}/WarmupStep${WARMUP_NUM}/"

export N_GPUS=1
export PARALLEL_ARGS="--pipefusion_parallel_degree 1 --ulysses_degree 1 --ring_degree 1"
echo "Running baseline"
bash ./benchmark/fid/generate.sh

export N_GPUS=4
export PARALLEL_ARGS="--pipefusion_parallel_degree 2 --ulysses_degree 1 --ring_degree 2"
TOKEN_MASK_OPTIONS=("Height" "Random" "EvenCluster" "UnevenCluster")
KV_MASK_OPTIONS=("Fixed" "Random" "MaxCosine" "MinCosine")
KV_MAX_OPTIONS=(64 56 48 40 32)
# TOKEN_MASK_OPTIONS=("Height")
# KV_MASK_OPTIONS=("Random")
# KV_MAX_OPTIONS=(64)
for token_mask in "${TOKEN_MASK_OPTIONS[@]}"; do
    for kv_mask in "${KV_MASK_OPTIONS[@]}"; do
        for kv_max in "${KV_MAX_OPTIONS[@]}"; do
            export TOKEN_MASK="--token_mask $token_mask"
            export KV_MASK="--kv_mask $kv_mask"
            export KV_MAX="--kv_max $kv_max"
            echo "Running with KV_MASK=$kv_mask, TOKEN_MASK=$token_mask, KV_MAX=$kv_max"
            bash ./benchmark/fid/generate.sh
        done
    done
done