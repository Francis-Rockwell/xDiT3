import torch
import torch.distributed
import json, os
from xfuser import xFuserStableDiffusion3Pipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_runtime_state,
)
import gc

def flush():
    gc.collect()
    torch.cuda.empty_cache()

def main():
    parser = FlexibleArgumentParser(description='xFuser Arguments')
    parser.add_argument('--caption_file', type=str, default='captions_coco.json')
    parser.add_argument('--sample_images_folder', type=str)
    parser.add_argument('--num_samples', type=int, default=30000)
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank

    pipe = xFuserStableDiffusion3Pipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.float16,
    ).to(f"cuda:{local_rank}")

    pipe.prepare_run(input_config)

    with open(args.caption_file) as f:
        raw_captions = json.load(f)

    num_samples = args.num_samples
    folder_path = args.sample_images_folder
    raw_captions = raw_captions['images'][:num_samples]
    captions = list(map(lambda x: x['sentences'][0]['raw'], raw_captions))
    filenames = list(map(lambda x: x['filename'], raw_captions))

    if engine_config.parallel_config.pp_degree == 1 and engine_config.parallel_config.sp_degree == 1:
        folder_path += "/original"
    else:
        folder_path += f"/WarmupStep{engine_config.runtime_config.warmup_steps}"
        folder_path += ("/KV" + str(engine_args.kv_max))
        folder_path += f"/{engine_config.runtime_config.kv_mask}KVMask"
        folder_path += f"/{engine_config.runtime_config.token_mask}TokenMask"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    num_prompt_one_step = 100
    for j in range(0, num_samples, num_prompt_one_step):
        output = pipe(
            height=input_config.height,
            width=input_config.width,
            prompt=captions[j:j+num_prompt_one_step],
            num_inference_steps=input_config.num_inference_steps,
            output_type=input_config.output_type,
            generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
        )
    
        if input_config.output_type == "pil":
            if pipe.is_dp_last_group():
                for k, local_filename in enumerate(filenames[j:j+num_prompt_one_step]):
                    output.images[k].save(f'{folder_path}/{local_filename}')
        print(f'{j}-{j+num_prompt_one_step-1} generation finished!')
        flush()        

    get_runtime_state().destory_distributed_env()


if __name__ == "__main__":
    main()
