#!/bin/bash

export SCRIPT="sd3_example.py"
export MODEL_ID="/archive/share/cql/model/stable-diffusion-3-medium-diffusers"
# export SCRIPT="flux_example.py"
# export MODEL_ID="/archive/share/cql/model/FLUX.1-dev"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export N_GPUS=1
export PARALLEL_ARGS="--pipefusion_parallel_degree 1 --ulysses_degree 1 --ring_degree 1"

export ITERATION="--iteration 1"
export INFERENCE_STEP="--num_inference_steps 50"
export WARMUP_STEP="--warmup_step 10"
export KV_MAX="--kv_max 1024"
export KV_MASK="--kv_mask Random"
export TOKEN_MASK="--token_mask Height"

bash ./examples/run.sh