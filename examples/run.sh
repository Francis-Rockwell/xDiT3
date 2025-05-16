set -x

export PYTHONPATH=$PWD:$PYTHONPATH

mkdir -p ./results

# task args
TASK_ARGS="--height 1024 --width 1024 --no_use_resolution_binning"

# CFG_ARGS="--use_cfg_parallel"

# By default, num_pipeline_patch = pipefusion_degree, and you can tune this parameter to achieve optimal performance.
# PIPEFUSION_ARGS="--num_pipeline_patch 8 "

# For high-resolution images, we use the latent output type to avoid runing the vae module. Used for measuring speed.
# OUTPUT_ARGS="--output_type latent"

# PARALLLEL_VAE="--use_parallel_vae"

# Another compile option is `--use_onediff` which will use onediff's compiler.
# COMPILE_FLAG="--use_torch_compile"

torchrun --nproc_per_node=$N_GPUS ./examples/$SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
$INFERENCE_STEP \
--prompt "A person is flying a kite on the lawn, and a large bird flies across the sky." \
$CFG_ARGS \
$PARALLLEL_VAE \
$COMPILE_FLAG \
$WARMUP_STEP \
$KV_MAX \
$TOKEN_MASK \
$KV_MASK \
$ITERATION \