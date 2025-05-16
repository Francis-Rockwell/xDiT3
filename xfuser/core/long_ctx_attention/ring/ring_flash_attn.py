import torch
from flash_attn.flash_attn_interface import _flash_attn_forward
from xfuser.core.long_ctx_attention import xFuserLongContextAttention
from xfuser.core.cache_manager.cache_manager import get_cache_manager
from yunchang.ring.utils import RingComm, update_out_and_lse
from yunchang.ring.ring_flash_attn import RingFlashAttnFunc
from xfuser.core.distributed import get_runtime_state


def xdit_ring_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    attn_layer=None,
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_strategy="none",
):
    supported_joint_strategy = ["none", "front", "rear"]
    if joint_strategy not in supported_joint_strategy:
        raise ValueError(
            f"joint_strategy: {joint_strategy} not supprted. supported joint strategy: {supported_joint_strategy}"
        )
    elif joint_strategy != "none" and (
        joint_tensor_key is None or joint_tensor_value is None
    ):
        raise ValueError(
            f"joint_tensor_key & joint_tensor_value must not be None when joint_strategy is not None"
        )

    comm = RingComm(process_group)

    out = None
    lse = None

    next_k, next_v = None, None

    if attn_layer is not None:
        if get_runtime_state().patch_mode and get_cache_manager().first_mask["attn", attn_layer]:
            k_cache, v_cache = get_cache_manager().get_kv_cache(layer=attn_layer)
            k_cache = k_cache.contiguous()
            v_cache = v_cache.contiguous()
            k_target_shape = get_runtime_state().get_target_shapes(k_cache, dim=1, token_shape=True, premask_split=True)
            v_target_shape = get_runtime_state().get_target_shapes(v_cache, dim=1, token_shape=True, premask_split=True)
            output_kv_cahce = [None]*comm.world_size
            current_kv_cache = [k_cache, v_cache]
            for step in range(comm.world_size):
                if step+1 != comm.world_size:
                    next_k_cache: torch.Tensor = comm.send_recv(to_send=k_cache, recv_tensor=k_target_shape[(comm.rank-1-step)%comm.world_size])
                    next_v_cache: torch.Tensor = comm.send_recv(to_send=v_cache, recv_tensor=v_target_shape[(comm.rank-1-step)%comm.world_size])
                    comm.commit()
                
                output_kv_cahce[(comm.rank-step)%comm.world_size] = current_kv_cache

                if step+1 != comm.world_size:
                    comm.wait()
                    current_kv_cache = [next_k_cache, next_v_cache]
            get_cache_manager().sync_cache(full_kv=output_kv_cahce, layer=attn_layer)
        k, v = get_cache_manager().update_and_get_kv_cache(
            new_kv=[k, v],
            layer=attn_layer,
            slice_dim=1,
            layer_type="attn",
        )
        k = k.contiguous()
        v = v.contiguous()

    k_target_shape = get_runtime_state().get_target_shapes(k, dim=1, token_shape=True)
    v_target_shape = get_runtime_state().get_target_shapes(v, dim=1, token_shape=True)

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            if get_runtime_state().patch_mode:
                to_send_k = get_runtime_state().masked_kv(k, (comm.rank-step)%comm.world_size)
                to_recv_k = get_runtime_state().masked_kv(k_target_shape[(comm.rank-1-step)%comm.world_size], (comm.rank-1-step)%comm.world_size)
                next_k: torch.Tensor = comm.send_recv(to_send=to_send_k, recv_tensor=to_recv_k)
                to_send_v = get_runtime_state().masked_kv(v, (comm.rank-step)%comm.world_size)
                to_recv_v = get_runtime_state().masked_kv(v_target_shape[(comm.rank-1-step)%comm.world_size], (comm.rank-1-step)%comm.world_size)
                next_v: torch.Tensor = comm.send_recv(to_send=to_send_v, recv_tensor=to_recv_v)
            else:
                next_k: torch.Tensor = comm.send_recv(to_send=k, recv_tensor=k_target_shape[(comm.rank-1-step)%comm.world_size])
                next_v: torch.Tensor = comm.send_recv(to_send=v, recv_tensor=v_target_shape[(comm.rank-1-step)%comm.world_size])
            comm.commit()

        if joint_strategy == "rear":
            if step + 1 == comm.world_size:
                key = torch.cat([k, joint_tensor_key], dim=1)
                value = torch.cat([v, joint_tensor_value], dim=1)
            else:
                key, value = k, v
        elif joint_strategy == "front":
            if step == 0:
                key = torch.cat([joint_tensor_key, k], dim=1)
                value = torch.cat([joint_tensor_value, v], dim=1)
            else:
                key, value = k, v
        elif joint_strategy == "none":
            key, value = k, v

        if not causal or step <= comm.rank:
            block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
                q,
                key,
                value,
                dropout_p,
                softmax_scale,
                causal=causal and step == 0,
                window_size=window_size,
                softcap=0.0,
                alibi_slopes=alibi_slopes,
                return_softmax=True and dropout_p > 0,
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


class xFuserRingFlashAttnFunc(RingFlashAttnFunc):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
        attn_layer,
        joint_tensor_key,
        joint_tensor_value,
        joint_strategy,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        if attn_layer is None:
            k = k.contiguous()
            v = v.contiguous()
        out, softmax_lse = xdit_ring_flash_attn_forward(
            group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
            attn_layer=attn_layer,
            joint_tensor_key=joint_tensor_key,
            joint_tensor_value=joint_tensor_value,
            joint_strategy=joint_strategy,
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        return out if not return_softmax else (out, softmax_lse, None)


def xdit_ring_flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    attn_layer=None,
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_strategy="none",
):
    return xFuserRingFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        attn_layer,
        joint_tensor_key,
        joint_tensor_value,
        joint_strategy,
    )
