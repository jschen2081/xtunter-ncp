import os
from typing import Tuple

import torch


def npu_token_permute(
    input_act: torch.Tensor,
    indices: torch.Tensor,
    num_topK: int | None = None,
    num_out_tokens: int | None = None,
    num_negative_one_in_indices: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if num_out_tokens is not None:
        raise NotImplementedError

    if num_negative_one_in_indices is not None:
        raise NotImplementedError

    if num_topK is not None:
        raise NotImplementedError

    from torch_npu import npu_moe_token_permute

    return npu_moe_token_permute(input_act, indices)


def npu_token_unpermute(
    input_act: torch.Tensor, row_id_map: torch.Tensor, probs: torch.Tensor | None = None
) -> torch.Tensor:
    from torch_npu import npu_moe_token_unpermute

    if probs is not None and not os.environ.get("XT_UNPERMUTE_FP32"):
        probs = probs.to(torch.bfloat16)
    # aclnn MoeTokenUnpermuteGrad 的 tiling 直接拒绝 0 行输入（EP_SIZE 较大时个别 rank 在个别层
    # 收不到任何 token）：前向算子容忍 0 行、反向算子不容忍，训练会在反向报 561002/EZ1008。
    # 0 行时改走等价的可微实现，绕开该算子；有 token 的路径与原来完全一致。
    if input_act.size(0) == 0 and os.environ.get("XT_UNPERMUTE_ZERO_SAFE", "1") != "0":
        if probs is None:
            # 输出行数与输入一致（同为 0 行），梯度可直连回 input_act
            return input_act
        num_unpermuted_tokens = probs.numel() // probs.size(-1)
        return torch.zeros(
            (num_unpermuted_tokens, input_act.size(-1)), dtype=input_act.dtype, device=input_act.device
        )
    return npu_moe_token_unpermute(input_act, row_id_map, probs=probs)
