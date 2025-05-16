#!/bin/bash

export SCRIPT="sd3_example.py"
export MODEL_ID="/archive/share/cql/model/stable-diffusion-3-medium-diffusers"
# export SCRIPT="flux_example.py"
# export MODEL_ID="/archive/share/cql/model/FLUX.1-dev"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export N_GPUS=4
export PARALLEL_ARGS="--pipefusion_parallel_degree 2 --ulysses_degree 1 --ring_degree 2"

export ITERATION="--iteration 1"
export INFERENCE_STEP="--num_inference_steps 50"
export WARMUP_STEP="--warmup_step 10"

TOKEN_MASK_OPTIONS=("Height" "Random" "EvenCluster" "UnevenCluster")
KV_MASK_OPTIONS=("Fixed" "Random" "MaxCosine" "MinCosine")
KV_MAX_OPTIONS=(1024 896 768 640 512)

for token_mask in "${TOKEN_MASK_OPTIONS[@]}"; do
    for kv_mask in "${KV_MASK_OPTIONS[@]}"; do
        for kv_max in "${KV_MAX_OPTIONS[@]}"; do
            export TOKEN_MASK="--token_mask $token_mask"
            export KV_MASK="--kv_mask $kv_mask"
            export KV_MAX="--kv_max $kv_max"
            echo "Running with KV_MASK=$kv_mask, TOKEN_MASK=$token_mask, KV_MAX=$kv_max"
            bash ./examples/run.sh
        done
    done
done
