import time
import os
import torch
import torch.distributed
from xfuser import xFuserStableDiffusion3Pipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    is_dp_last_group,
    get_data_parallel_rank,
    get_runtime_state,
)
from xfuser.core.distributed.parallel_state import get_data_parallel_world_size


def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    parser.add_argument('--iteration', type=int, default=10)
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank
    pipe = xFuserStableDiffusion3Pipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.float16,
    ).to(f"cuda:{local_rank}")

    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_tp{engine_args.tensor_parallel_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}_"
        f"{'KV'+str(engine_args.kv_max)}_{engine_config.runtime_config.kv_mask}KVMask_{engine_config.runtime_config.token_mask}TokenMask_"
    )

    directory_path = f"./results/SD3/TotalStep{input_config.num_inference_steps}/WarmupStep{engine_config.runtime_config.warmup_steps}/KV{engine_args.kv_max}"

    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    pipe.prepare_run(input_config)

    latency = []
    for j in range(args.iteration):
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        output = pipe(
            height=input_config.height,
            width=input_config.width,
            prompt=input_config.prompt,
            num_inference_steps=input_config.num_inference_steps,
            output_type=input_config.output_type,
            generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

        if j+1 == args.iteration and input_config.output_type == "pil":
            dp_group_index = get_data_parallel_rank()
            num_dp_groups = get_data_parallel_world_size()
            dp_batch_size = (input_config.batch_size + num_dp_groups - 1) // num_dp_groups
            if pipe.is_dp_last_group():
                if not os.path.exists("results"):
                    os.mkdir("results")
                for i, image in enumerate(output.images):
                    image_rank = dp_group_index * dp_batch_size + i
                    os.makedirs(directory_path, exist_ok=True)
                    image.save(
                        f"{directory_path}/{parallel_info}{image_rank}.png"
                    )
                    print(
                        f"image {i} saved to {directory_path}/{parallel_info}{image_rank}.png"
                    )

        if get_world_group().rank == get_world_group().world_size - 1:
            latency.append(elapsed_time)
            print(f"epoch time: {elapsed_time:.4f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, peak memory: {peak_memory/1e9:.2f} GB")
            if j+1 == args.iteration:
                filtered_latency = latency
                print(f"Average epoch time: {(sum(filtered_latency)/len(filtered_latency)):.4f} sec")

    get_runtime_state().destory_distributed_env()


if __name__ == "__main__":
    main()
