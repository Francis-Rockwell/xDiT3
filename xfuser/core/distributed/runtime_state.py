from abc import ABCMeta
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from diffusers import DiffusionPipeline, CogVideoXPipeline
import torch.distributed

from xfuser.config.config import (
    ParallelConfig,
    RuntimeConfig,
    InputConfig,
    EngineConfig,
)
from xfuser.logger import init_logger
from .parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_pp_group,
    get_sp_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_pipeline_parallel_rank,
    get_pipeline_parallel_world_size,
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
    is_pipeline_last_stage,
)

from .utils import AdaptedKMeans

from .runtime_state_core import DiTRuntimeStateCore

logger = init_logger(__name__)


class DiTRuntimeState(DiTRuntimeStateCore):
    num_pipeline_patch: int
    pipeline_group_idx: int
    num_sequence_patch: int
    sequence_group_idx: int

    latents_height: int
    latents_width: int
    token_height: int
    token_width: int
    
    premask_pp_patches_height: Optional[List[int]]

    sp_pp_patches_height: Optional[List[List[int]]]
    premask_sp_pp_patches_height: Optional[List[List[int]]]

    height_to_token_factor: float
    premask_height_to_token_factor: float

    sp_patches_height: Optional[List[int]]
    premask_sp_patches_height: Optional[List[int]]

    sp_patches_start_end_idx_local: Optional[List[List[int]]]
    premask_sp_patches_start_end_idx_local: Optional[List[List[int]]]

    token_mask_sp_pp_1d: Optional[torch.Tensor]
    kv_masks: Optional[List[torch.Tensor]]

    premask_hidden_state: Optional[torch.Tensor]
    premask_hidden_state_full: Optional[torch.Tensor]

    device: torch.device
    gpu_generator: torch.Generator
    cpu_generator: torch.Generator
    
    def __init__(self, pipeline: DiffusionPipeline, config: EngineConfig):
        super().__init__(pipeline, config)
        
        self.num_pipeline_patch = get_pipeline_parallel_world_size()
        self.pipeline_group_idx = get_pipeline_parallel_rank()
        self.num_sequence_patch = get_sequence_parallel_world_size()
        self.sequence_group_idx = get_sequence_parallel_rank()

        self.premask_hidden_state_full = None
        self.device = get_pp_group().device
        self.gpu_generator = torch.Generator(self.device).manual_seed(42)
        self.cpu_generator = torch.Generator("cpu").manual_seed(42)
        
    def _calc_patches_metadata(self):
        patch_size = self.backbone_patch_size
        self.latents_height = self.input_config.height // self.vae_scale_factor
        self.latents_width = self.input_config.width // self.vae_scale_factor
        self.token_height = self.latents_height // patch_size
        self.token_width = self.latents_width // patch_size

        pp_sp_patches_height, pp_sp_patches_start_idx = self.generate_pp_sp_patches_data()

        self.pp_patches_height = [sp_patches_height[self.sequence_group_idx] for sp_patches_height in pp_sp_patches_height]
        self.pp_patches_start_idx_local = [0] + [sum(self.pp_patches_height[:i]) for i in range(1, len(self.pp_patches_height) + 1)]
        self.pp_patches_start_end_idx_global = [
            sp_patches_start_idx[self.sequence_group_idx : self.sequence_group_idx + 2]
            for sp_patches_start_idx in pp_sp_patches_start_idx
        ]
        self.pp_patches_token_start_end_idx_global = [
            [(self.token_width) * (start_idx // patch_size), (self.token_width) * (end_idx // patch_size)]
            for start_idx, end_idx in self.pp_patches_start_end_idx_global
        ]
        self.pp_patches_token_num = [end - start for start, end in self.pp_patches_token_start_end_idx_global]
        self.pp_patches_token_start_idx_local = [sum(self.pp_patches_token_num[:i]) for i in range(len(self.pp_patches_token_num) + 1)]
        
        self.height_to_token_factor = self.token_width // patch_size
        self.sp_pp_patches_height = [[pp_patches_height[sp_idx] for pp_patches_height in pp_sp_patches_height] for sp_idx in range(self.num_sequence_patch)]
        self.sp_patches_height = [sum(pp_patches_height) for pp_patches_height in self.sp_pp_patches_height]
        sp_patches_start_end_idx_local = []
        for sp_idx in range(self.num_sequence_patch):
            sp_patches_start_idx_local = [0] + [sum(self.sp_pp_patches_height[sp_idx][:i]) for i in range(1, self.num_sequence_patch + 1)]
            sp_patches_start_end_idx_local.append([sp_patches_start_idx_local[pp_idx : pp_idx+2] for pp_idx in range(self.num_pipeline_patch)])
        self.sp_patches_start_end_idx_local = sp_patches_start_end_idx_local
    
    def random_token_mask(self):
        labels = torch.randint(
                    0,
                    self.num_sequence_patch*self.num_pipeline_patch,
                    (self.token_height, self.token_width),
                    generator=self.gpu_generator,
                    device=self.device)
        token_mask_sp_pp_2d = torch.stack([(labels == i) for i in range(self.num_sequence_patch * self.num_pipeline_patch)]) \
                                    .reshape(self.num_sequence_patch, self.num_pipeline_patch, -1)

        return token_mask_sp_pp_2d
    
    def height_token_mask(self):
        token_mask_sp_pp_2d = torch.zeros(
                                size=(self.num_sequence_patch, self.num_pipeline_patch, self.token_height, self.token_width),
                                dtype=bool,
                                device=self.device)
        h = self.token_height // (self.num_sequence_patch * self.num_pipeline_patch)
        for i in range(self.num_pipeline_patch):
            for j in range(self.num_sequence_patch):
                start = (i*self.num_sequence_patch+j)*h
                end = start+h
                token_mask_sp_pp_2d[j, i, start:end, :] = True
        return token_mask_sp_pp_2d

    def cluster_token_mask(self, even:bool):
        token_mask_sp_pp_2d = torch.zeros(
                                size=(self.num_sequence_patch, self.num_pipeline_patch, self.token_height, self.token_width),
                                dtype=bool,
                                device=self.device)
        if is_pipeline_last_stage():
            self.gather_hidden_state()
            num_cluster = self.num_sequence_patch * self.num_pipeline_patch
            kmeans = AdaptedKMeans(n_clusters=num_cluster, even=even)
            kmeans.fit(self.premask_hidden_state_full.cpu().float().numpy())
            labels = kmeans.labels_.reshape(self.token_height, self.token_width)
            token_mask_sp_pp_2d = torch.stack([torch.tensor(labels==i) for i in range(num_cluster)]) \
                                        .reshape(self.num_sequence_patch, self.num_pipeline_patch, -1) \
                                        .to(self.device)
        get_pp_group().broadcast(token_mask_sp_pp_2d, self.num_pipeline_patch-1)
        get_pp_group().barrier()
        return token_mask_sp_pp_2d
    
    def gather_hidden_state(self):
        sp_hidden_states_list = get_sp_group().all_gather(self.premask_hidden_state.contiguous(), separate_tensors=True)
        hidden_states_list = []
        slice_token_num = self.token_height * self.token_width // self.num_pipeline_patch // self.num_sequence_patch
        for pp_idx in range(self.num_pipeline_patch):
            hidden_states_list += [
                sp_hidden_states_list[sp_idx][
                    slice_token_num*pp_idx : slice_token_num*(pp_idx+1)
                    :,
                ]
                for sp_idx in range(self.num_sequence_patch)
            ]
        hidden_state_full = torch.cat(hidden_states_list, dim=-2)
        self.premask_hidden_state_full = torch.nn.functional.normalize(hidden_state_full, p=2, dim=1)
    
    def set_token_mask(self, state:bool = True):
        if state:
            match self.runtime_config.token_mask.lower():
                case "evencluster":
                    token_mask_sp_pp_2d = self.cluster_token_mask(even=True)
                case "unevencluster":
                    token_mask_sp_pp_2d = self.cluster_token_mask(even=False)
                case "random":
                    token_mask_sp_pp_2d = self.random_token_mask()
                case "height":
                    token_mask_sp_pp_2d = self.height_token_mask()
                case _:
                    raise NotImplementedError("No such token mask")

            pp_sp_patches_height = [
                [token_mask_sp_pp_2d[sp_idx][pp_idx].sum().item()*self.backbone_patch_size for sp_idx in range(self.num_sequence_patch)]
                for pp_idx in range(self.num_pipeline_patch)
            ]

            self.token_mask_sp_pp_1d = token_mask_sp_pp_2d.view(self.num_sequence_patch, self.num_pipeline_patch, -1)

            self._recalc_patches_metadata(pp_sp_patches_height)
        else:
            self._calc_patches_metadata()
            self._reset_recv_buffer()

    def _recalc_patches_metadata(self, pp_sp_patches_height):     
        # pp_patches_height is needed in transformer
        pp_patches_height = [sp_patches_height[self.sequence_group_idx] for sp_patches_height in pp_sp_patches_height]
        self.premask_pp_patches_height = self.pp_patches_height
        self.pp_patches_height = pp_patches_height
        
        # height_to_token_factor is needed to generate target shapes
        self.premask_height_to_token_factor = self.height_to_token_factor
        self.height_to_token_factor = 1 / self.backbone_patch_size

        self.premask_sp_pp_patches_height = self.sp_pp_patches_height
        self.sp_pp_patches_height = [[pp_patches_height[sp_idx] for pp_patches_height in pp_sp_patches_height] for sp_idx in range(self.num_sequence_patch)]

        # sp_patches_height is needed to generate target shapes
        self.premask_sp_patches_height = self.sp_patches_height
        self.sp_patches_height = [sum(pp_patches_height) for pp_patches_height in self.sp_pp_patches_height]

        # sp_patches_start_end_idx_local is needed to reorder latents
        self.premask_sp_patches_start_end_idx_local = self.sp_patches_start_end_idx_local
        sp_patches_start_end_idx_local = []
        for sp_idx in range(self.num_sequence_patch):
            sp_patches_start_idx_local = [0] + [sum(self.sp_pp_patches_height[sp_idx][:i]) for i in range(1, self.num_sequence_patch + 1)]
            sp_patches_start_end_idx_local.append([sp_patches_start_idx_local[pp_idx : pp_idx+2] for pp_idx in range(self.num_pipeline_patch)])
        self.sp_patches_start_end_idx_local = sp_patches_start_end_idx_local

        self.pp_patches_start_idx_local = None
        self.pp_patches_start_end_idx_global = None
        self.pp_patches_token_start_idx_local = None
        self.pp_patches_token_start_end_idx_global = None
        self.pp_patches_token_num = None

    def get_target_shapes(self, tensor: torch.Tensor, dim: int, token_shape: bool=False, premask_split: bool=False):
        target_shapes = []
        heights = self.premask_sp_patches_height if premask_split else self.sp_patches_height
        height_to_token_factor = self.premask_height_to_token_factor if premask_split else self.height_to_token_factor
        for height in heights:
            size = list(tensor.size())
            size[dim] = int(height * (height_to_token_factor if token_shape else 1))
            target_shapes.append(torch.empty(size=size, dtype=tensor.dtype, layout=tensor.layout, device=tensor.device))
        return target_shapes

    def masked_latents(self, latents: torch.Tensor):
        patch_size = self.backbone_patch_size
        latents = latents.reshape(
            shape=(
                latents.shape[0],
                latents.shape[1],
                self.token_height,
                patch_size,
                self.token_width,
                patch_size
            )
        )
        latents = torch.einsum("nchpwq->nchwpq", latents)
        latents = latents.reshape(
            shape=(
                latents.shape[0],
                latents.shape[1],
                self.token_height * self.token_width,
                patch_size,
                patch_size
            )
        )
        latents_list = [latents[:, :, mask, ...] for mask in self.token_mask_sp_pp_1d[self.sequence_group_idx]]
        for i, latent in enumerate(latents_list):
            latents_list[i] = latent.reshape(
                shape=(
                    latents.shape[0],
                    latents.shape[1],
                    -1,
                    patch_size
                )
            )
        return latents_list

    def reshape_masked(self, latents_list: list[torch.Tensor]):

        mask = self.token_mask_sp_pp_1d
        mask = torch.einsum("spl->lps", mask)
        mask = mask.reshape(shape=(mask.shape[0], -1))

        tmp_latents_list = [latent.reshape(
            shape=(
                latent.shape[0],
                latent.shape[1],
                -1,
                1,
                self.backbone_patch_size, 
                self.backbone_patch_size,
            )
        ) for latent in latents_list]

        new_latents_list = []
        indice = [0]*mask.shape[1]
        for m in mask:
            mask_index = m.nonzero(as_tuple=True)[0].item()
            new_latents_list.append(tmp_latents_list[mask_index][:, :, indice[mask_index], 0:1, ...])
            indice[mask_index] += 1
        
        latent = torch.cat(new_latents_list, dim=2)
        latent = latent.reshape(
            shape=(
                latent.shape[0],
                latent.shape[1],
                -1,
                self.token_width,
                self.backbone_patch_size, 
                self.backbone_patch_size, 
            )
        )
        latent = torch.einsum("nchwpq->nchpwq", latent)
        latent = latent.reshape(
            shape=(
                latent.shape[0],
                latent.shape[1],
                -1,
                self.latents_width
            )
        )
        return latent

    def sp_latents_all_gather(self, latents: torch.Tensor, reshape_masked: bool = False):
        latents = latents.contiguous()
        target_shapes = self.get_target_shapes(latents, dim=-2)
        sp_latents_list = get_sp_group().all_gather(latents, separate_tensors=True, target_shapes=target_shapes)
        latents_list = []
        for pp_idx in range(self.num_pipeline_patch):
            for sp_idx in range(self.num_sequence_patch):
                latents_list.append(
                    sp_latents_list[sp_idx][
                        ...,
                        self.sp_patches_start_end_idx_local[sp_idx][pp_idx][0] : self.sp_patches_start_end_idx_local[sp_idx][pp_idx][1],
                        :,
                    ]
                )
        if reshape_masked:
            return self.reshape_masked(latents_list)
        else:
            return torch.cat(latents_list, dim=-2)

    def random_kv_mask(self, permute: bool):
        sp_pp_token_num = [
            [int(height * self.height_to_token_factor) for height in sp_patches_height]
            for sp_patches_height in self.sp_pp_patches_height
        ]

        self.kv_masks = []
        for pp_token_num in sp_pp_token_num:
            pp_masks = []
            for token_num in pp_token_num:
                valid_token_num = min(token_num, self.runtime_config.kv_max)
                ones = torch.ones(size=(valid_token_num,), dtype=bool, device=self.device)
                zeros = torch.zeros(size=(token_num-valid_token_num,), dtype=bool, device=self.device)
                mask = torch.cat((ones, zeros))
                if permute:
                    mask = mask[torch.randperm(token_num, generator=self.cpu_generator)]
                pp_masks.append(mask)
            self.kv_masks.append(torch.cat(pp_masks))
    
    def cosine_kv_mask(self, largest: bool):
        kv_masks = torch.zeros(
                            size=(self.num_sequence_patch, self.token_mask_sp_pp_1d.shape[-1]),
                            dtype=torch.bool,
                            device=self.device)
        if is_pipeline_last_stage():
            kv_mask_list = []        
            for sp_idx in range(self.num_sequence_patch):
                mask = torch.any(self.token_mask_sp_pp_1d[sp_idx], dim=0)
                if self.premask_hidden_state_full is None:
                    self.gather_hidden_state()
                cosine_sim = torch.matmul(self.premask_hidden_state_full[mask], self.premask_hidden_state_full[~mask].mean(dim=0))
                _, indices = torch.topk(cosine_sim, min(self.runtime_config.kv_max * self.num_pipeline_patch, cosine_sim.shape[0]), largest=largest)
                kv_mask = torch.zeros(self.token_mask_sp_pp_1d.shape[-1], dtype=torch.bool, device=self.device)
                kv_mask[indices] = True
                kv_mask_list.append(kv_mask)
            kv_masks = torch.stack(kv_mask_list)
        get_pp_group().broadcast(kv_masks, self.num_pipeline_patch-1)
        get_pp_group().barrier()
        self.kv_masks = []
        for sp_idx in range(self.num_sequence_patch):
            token_num = sum(torch.any(self.token_mask_sp_pp_1d[sp_idx], dim=0))
            self.kv_masks.append(kv_masks[sp_idx][:token_num])

    def set_kv_mask(self):
        match self.runtime_config.kv_mask.lower():
            case "fixed":
                self.random_kv_mask(permute=False)
            case "random":
                self.random_kv_mask(permute=True)
            case "maxcosine":
                self.cosine_kv_mask(largest=True)
            case "mincosine":
                self.cosine_kv_mask(largest=False)
            case _:
                raise NotImplementedError("No such kv mask")
            
    def masked_kv(self, kv: torch.Tensor, sp_idx: int):
        return kv[:, self.kv_masks[sp_idx], ...].contiguous()

# _RUNTIME: Optional[RuntimeState] = None
# TODO: change to RuntimeState after implementing the unet
_RUNTIME: Optional[DiTRuntimeState] = None


def runtime_state_is_initialized():
    return _RUNTIME is not None


def get_runtime_state():
    assert _RUNTIME is not None, "Runtime state has not been initialized."
    return _RUNTIME


def initialize_runtime_state(pipeline: DiffusionPipeline, engine_config: EngineConfig):
    global _RUNTIME
    if _RUNTIME is not None:
        logger.warning(
            "Runtime state is already initialized, reinitializing with pipeline..."
        )
    if hasattr(pipeline, "transformer"):
        _RUNTIME = DiTRuntimeState(pipeline=pipeline, config=engine_config)
