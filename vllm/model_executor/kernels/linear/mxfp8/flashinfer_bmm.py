# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_VALUE_DTYPE,
    mxfp8_e4m3_quantize,
    swizzle_mxfp8_scale,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_bmm_mxfp8
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import direct_register_custom_op

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig

# FlashInfer's F8_128x4 scale swizzle tiles 128 rows. Its bmm_mxfp8 reads the
# scales of each batch as a separate 128-row-padded block, but mxfp8_quantize
# pads the flattened batch * rows dimension instead, so batches only separate
# when every batch has a multiple of 128 rows (flashinfer-ai/flashinfer#3604).
_SF_ROW_TILE = 128


class FlashInferMxfp8BmmLinearKernel(Mxfp8LinearKernel):
    """Grouped MXFP8 BMM via FlashInfer ``bmm_mxfp8`` (cuDNN backend).

    Covers MXFP8 BMM layers (``is_bmm``) when DeepGEMM is unavailable or
    disabled, e.g. on SM107 with ``VLLM_USE_DEEP_GEMM=0``. Weights stay FP8 as
    ``[groups, N, K]`` with F8_128x4 swizzled E8M0 scales.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
        ):
            return False, "FlashInfer MXFP8 BMM is enabled for SM10x only."
        if not has_flashinfer_bmm_mxfp8():
            return False, (
                "requires FlashInfer bmm_mxfp8 with a cuDNN backend that "
                "supports this GPU."
            )
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        groups = c.bmm_batch_size
        if groups is None or groups <= 0:
            return False, "FlashInfer MXFP8 BMM requires a positive batch size."
        total_n, k = c.weight_shape
        if total_n % groups != 0:
            return False, f"N={total_n} is not divisible by {groups} groups."
        n = total_n // groups
        # bmm_mxfp8 requires n, k >= 128. A multiple of 128 rows per group also
        # keeps the flattened weight-scale swizzle per-group separable.
        if n < _SF_ROW_TILE or n % _SF_ROW_TILE != 0:
            return False, f"per-group N={n} must be a positive multiple of 128."
        if k < 128 or k % MXFP8_BLOCK_SIZE != 0:
            return False, f"K={k} must be >= 128 and divisible by 32."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        if weight.ndim == 3:
            return
        assert weight.dtype == MXFP8_VALUE_DTYPE and weight.ndim == 2
        groups = self.config.bmm_batch_size
        assert groups is not None
        total_n, k = weight.shape
        scale = layer.weight_scale.data[:total_n, : k // MXFP8_BLOCK_SIZE]
        if scale.dtype != torch.uint8:
            # DeepSeek checkpoints store E8M0 scales as float8_e8m0fnu.
            scale = scale.view(torch.uint8)
        # Per-group N is a multiple of 128 (can_implement), so swizzling the
        # flattened [groups * N, K / 32] scales equals per-group swizzling.
        scale = swizzle_mxfp8_scale(scale.contiguous(), M=total_n, K=k)
        replace_parameter(
            layer, "weight", weight.contiguous().view(groups, total_n // groups, k)
        )
        replace_parameter(layer, "weight_scale", scale.contiguous())

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project a [tokens, groups, K] activation to [tokens, groups, N]."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                "FlashInfer MXFP8 BMM quantizes its own activation; a "
                "pre-quantized activation is not supported."
            )
        output = torch.ops.vllm.flashinfer_mxfp8_bmm(
            x, layer.weight, layer.weight_scale
        )
        if bias is not None:
            output = output + bias.view(output.shape[1:])
        return output


def _flashinfer_mxfp8_bmm(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    from flashinfer import bmm_mxfp8

    tokens, groups, k = x.shape
    n = weight.shape[1]
    if tokens == 0:
        return x.new_empty((0, groups, n), dtype=torch.bfloat16)
    padded = round_up(tokens, _SF_ROW_TILE)
    x_gmk = x.transpose(0, 1)
    if padded != tokens:
        x_gmk = F.pad(x_gmk, (0, 0, 0, padded - tokens))
    # Quantize as [groups * padded, K]: each group then owns whole 128-row
    # scale tiles, which bmm_mxfp8 needs for groups > 1 (flashinfer#3604).
    x_q, x_scale = mxfp8_e4m3_quantize(
        x_gmk.reshape(groups * padded, k), is_sf_swizzled_layout=True
    )
    out = bmm_mxfp8(
        x_q.view(groups, padded, k),
        weight.transpose(-2, -1),
        x_scale,
        weight_scale,
        torch.bfloat16,
        backend="cudnn",
    )
    return out[:, :tokens].transpose(0, 1).contiguous()


def _flashinfer_mxfp8_bmm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], weight.shape[0], weight.shape[1]),
        device=x.device,
        dtype=torch.bfloat16,
    )


direct_register_custom_op(
    "flashinfer_mxfp8_bmm",
    _flashinfer_mxfp8_bmm,
    fake_impl=_flashinfer_mxfp8_bmm_fake,
)
