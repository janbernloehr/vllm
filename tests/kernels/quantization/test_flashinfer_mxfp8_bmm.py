# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer MXFP8 BMM kernel: selection (CPU) and numerics (SM10x GPU).

Run `pytest tests/kernels/quantization/test_flashinfer_mxfp8_bmm.py`.
"""

from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    DeepGemmMxfp8BmmLinearKernel,
    EmulationMxfp8LinearKernel,
    FlashInferMxfp8BmmLinearKernel,
    Mxfp8LinearLayerConfig,
    init_mxfp8_linear_kernel,
)
from vllm.platforms import current_platform

# DeepSeek-V4.1-Flash wo_a at TP4: 2 local groups of (1024, 4096).
_DSV41_TP4 = dict(weight_shape=(2048, 4096), bmm_batch_size=2)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("weight_shape", "groups", "ok"),
    [
        ((2048, 4096), 2, True),
        ((1024, 4096), 8, True),
        ((2048, 4096), None, False),
        ((2048, 4096), 3, False),  # N not divisible by groups
        ((192, 4096), 2, False),  # per-group N=96 < 128
        ((384, 4096), 2, False),  # per-group N=192 is not a multiple of 128
        ((2048, 64), 2, False),  # K < 128
    ],
)
def test_flashinfer_mxfp8_bmm_can_implement(weight_shape, groups, ok):
    config = Mxfp8LinearLayerConfig(weight_shape=weight_shape, bmm_batch_size=groups)
    can_implement, reason = FlashInferMxfp8BmmLinearKernel.can_implement(config)
    assert can_implement == ok, reason


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("deep_gemm", "flashinfer", "expected"),
    [
        (True, True, DeepGemmMxfp8BmmLinearKernel),
        (False, True, FlashInferMxfp8BmmLinearKernel),
        (False, False, EmulationMxfp8LinearKernel),
    ],
)
def test_bmm_selection_prefers_flashinfer_over_emulation(
    deep_gemm, flashinfer, expected
):
    """With DeepGEMM unavailable or disabled (VLLM_USE_DEEP_GEMM=0), an MXFP8
    BMM layer must use FlashInfer before falling back to emulation."""
    with (
        patch("vllm.model_executor.kernels.linear.current_platform") as platform,
        patch.object(
            DeepGemmMxfp8BmmLinearKernel,
            "is_supported",
            return_value=(deep_gemm, None if deep_gemm else "no DeepGEMM"),
        ),
        patch.object(
            FlashInferMxfp8BmmLinearKernel,
            "is_supported",
            return_value=(flashinfer, None if flashinfer else "no FlashInfer"),
        ),
    ):
        platform.is_cuda.return_value = True
        kernel = init_mxfp8_linear_kernel(**_DSV41_TP4)
    assert type(kernel) is expected


def _flashinfer_bmm_available() -> bool:
    return (
        current_platform.is_cuda() and FlashInferMxfp8BmmLinearKernel.is_supported()[0]
    )


@pytest.mark.skipif(
    not _flashinfer_bmm_available(),
    reason="requires an SM10x GPU and FlashInfer cuDNN bmm_mxfp8",
)
@pytest.mark.parametrize("num_tokens", [1, 7, 128, 300])
@pytest.mark.parametrize(("groups", "n", "k"), [(2, 1024, 4096), (1, 128, 256)])
def test_flashinfer_mxfp8_bmm_matches_dequantized_reference(num_tokens, groups, n, k):
    """Tokens that are not a multiple of 128 with groups > 1 cover
    flashinfer-ai/flashinfer#3604, which the kernel pads around.

    Inputs and the reference are built on the CPU so that only the kernel under
    test runs on the GPU: elementwise ops such as torch.exp2 can JIT-compile
    through NVRTC on CUDA, which fails on GPUs newer than the bundled NVRTC
    (e.g. sm_107)."""
    torch.manual_seed(0)
    weight_cpu = (torch.randn(groups * n, k) * 0.5).to(torch.float8_e4m3fn)
    scales_cpu = torch.randint(122, 131, (groups * n, k // 32), dtype=torch.uint8)
    reference_weight = (
        weight_cpu.float().view(groups * n, k // 32, 32)
        * torch.exp2(scales_cpu.float() - 127)[..., None]
    ).view(groups, n, k)

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight_cpu.cuda(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scales_cpu.cuda(), requires_grad=False)
    kernel = FlashInferMxfp8BmmLinearKernel(
        Mxfp8LinearLayerConfig(weight_shape=(groups * n, k), bmm_batch_size=groups)
    )
    kernel.process_weights_after_loading(layer)
    assert layer.weight.shape == (groups, n, k)
    assert layer.weight.dtype == torch.float8_e4m3fn

    x_cpu = torch.randn(num_tokens, groups, k).to(torch.bfloat16)
    output = kernel.apply_weights(layer, x_cpu.cuda())
    assert output.shape == (num_tokens, groups, n)
    assert output.dtype == torch.bfloat16
    output = output.cpu().float()
    reference = torch.einsum("tgk,gnk->tgn", x_cpu.float(), reference_weight)
    assert torch.isfinite(output).all()
    rel_err = (output - reference).norm() / reference.norm()
    assert rel_err < 0.06, rel_err
    # Per group, so a corrupted group > 0 (flashinfer#3604) cannot hide.
    for g in range(groups):
        cos = torch.nn.functional.cosine_similarity(
            output[:, g].flatten(), reference[:, g].flatten(), dim=0
        )
        assert cos > 0.99, (g, cos)
