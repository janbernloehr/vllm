<!-- markdownlint-disable -->

## Overview

Add FlashInfer's cuDNN `bmm_mxfp8` as a functional MXFP8 BMM fallback when DeepGEMM is unavailable or disabled. This avoids DeepSeek-V4.1-Flash NVFP4's emulation startup failure on SM107; DeepGEMM remains preferred.

Related hardware tracking: [#49735](https://github.com/vllm-project/vllm/issues/49735).

## Claims

- Select MXFP8 BMM kernels in this order: DeepGEMM → FlashInfer → Emulation. Non-BMM selection is unchanged.
- Gate `FlashInferMxfp8BmmLinearKernel` on SM10x, FlashInfer's cuDNN backend support, and compatible grouped weight dimensions.
- Route DeepSeek-V4/V4.1 `wo_a` through its selected FlashInfer kernel using BF16 inverse-RoPE output. Exclude FlashInfer weights from fused WO-A, and disable FlashMLA mega-attention when DeepGEMM is unavailable.
- Preserve per-group token padding to a multiple of 128 for the cuDNN scale-layout issue [flashinfer#3604](https://github.com/flashinfer-ai/flashinfer/issues/3604).

## Validation

**Historical GPU validation of the supplied patch, before this branch was prepared.** On 2026-10-17: vLLM `49cb2bbf71cfd36ee5aa33d1e184b1cddff55294`, `vllm/vllm-openai:cu134-nightly` from that day, FlashInfer 0.7.2, PyTorch 2.14.1, 4 × SM107 GPUs, TP4. Model: `nvidia/DeepSeek-V4.1-Flash-NVFP4`; `--quantization modelopt_mixed --tokenizer-mode deepseek_v41 --kv-cache-dtype fp8`. Workload: 8K input / 1K output.

The unpatched control and patched treatment both used `VLLM_USE_DEEP_GEMM=0`, without linear-backend overrides.

| Configuration | Selected MXFP8 BMM kernel | Result |
|---|---|---|
| Unpatched, DeepGEMM off | `EmulationMxfp8LinearKernel` | Weight loading fails in `torch.exp2` with NVRTC rejecting `sm_107` |
| Patched, DeepGEMM off | `FlashInferMxfp8BmmLinearKernel` | 18/18 tests pass; server starts and benchmark completes without emulation or NVRTC errors |
| DeepGEMM-enabled reference | `DeepGemmMxfp8BmmLinearKernel` | Server starts and benchmark completes |

| Concurrency | FlashInfer configuration output tok/s | DeepGEMM reference output tok/s |
|---|---:|---:|
| 1 | 193.2 | 250.1 |
| 64 | 4752.7 | 5352.8 |
| 256 | 7895.2 | 11233.4 |

These measurements compare whole configurations: disabling DeepGEMM affects other layers and excludes optimized attention/projection paths. They do not measure `bmm_mxfp8` alone. This is a functional fallback, with no speedup claim. No repeated-run variability or per-kernel profile was collected.

The same change previously passed 18/18 tests and served successfully on vLLM `7d0b4e57a` with FlashInfer 0.7.0.post1.

Kernel test command used for the historical GPU validation:

```bash
.venv/bin/python -m pytest tests/kernels/quantization/test_flashinfer_mxfp8_bmm.py -v
```

The eight GPU cases compare against a CPU-computed dequantized reference: grouped shapes `2 × 1024 × 4096` and `1 × 128 × 256`, each with 1, 7, 128 and 300 tokens. They require relative error < 0.06 and per-group cosine similarity > 0.99. The other ten cases check shape rejection and DeepGEMM → FlashInfer → Emulation selection.

**Checks on the 2026-10-20 refresh.** Rebased onto upstream main `e4a676cdc22f99e808093a87ff8509a6089d392d`. The validated kernel, numerics tests and changed integration/helper functions retain the same Python AST. The rebase preserves upstream's newer mega-attention implementation; the only conflict was adjacent imports, resolved by keeping both imports. The branch also wires the numerics test into B200 CI. CPU selection/shape tests passed again: 49 passed, with the eight GPU cases deselected on this macOS host. Fresh SM107 reproduction, GPU kernel tests and end-to-end model accuracy evaluation are pending. The historical results above are not a GPU run of this refreshed branch and do not include a model accuracy evaluation.

```bash
.venv/bin/python -m pytest --noconftest \
  tests/kernels/quantization/test_flashinfer_mxfp8_bmm.py \
  tests/kernels/quantization/test_mxfp8_kernel_selection.py -m cpu_test -v
```

These CPU tests do not use shared fixtures; `--noconftest` avoids importing the model/GPU test fixture stack for this local run.

Example startup reproduction in a compatible CUDA environment (install the checkout first):

```bash
VLLM_USE_DEEP_GEMM=0 vllm serve nvidia/DeepSeek-V4.1-Flash-NVFP4 \
  --tensor-parallel-size 4 --quantization modelopt_mixed \
  --tokenizer-mode deepseek_v41 --kv-cache-dtype fp8
```

For benchmark reproduction, send 8192-token prompts with 1024-token outputs at concurrency 1, 64 and 256 using `vllm bench serve`. The table reports the historical runs; the startup command above is a minimal reproduction, not a record of every benchmark tuning option.

## Details

- **Root cause:** MXFP8 BMM selection previously tried DeepGEMM and then emulation. The latter dequantizes weights with `torch.exp2`; in the tested PyTorch/CUDA toolchain, jiterator/NVRTC rejects `sm_107`. This change avoids that path. It does not fix PyTorch/CUDA jiterator's architecture support.
- **Layout and limits:** weights remain FP8 `[groups, N, K]` with F8_128x4-swizzled E8M0 scales. Per-group N must be a positive multiple of 128; K must be at least 128 and divisible by 32. The kernel owns activation quantization and returns BF16 `[tokens, groups, N]`.
- **Padding:** quantize `[groups * padded_tokens, K]` only after padding each group's tokens to 128 rows. Without this, scale tiles can straddle groups and corrupt groups after the first. Keep this workaround while the cuDNN path of [flashinfer#3604](https://github.com/flashinfer-ai/flashinfer/issues/3604) remains unfixed.
- **Design and tradeoffs:** reuse the linear-kernel registry, MXFP8 quantization/swizzle helpers and custom-op fake registration. Changes are confined to the kernel, selection, capability detection and DeepSeek projection/attention gates. There are no model-runner or scheduler changes. Token padding adds work for small batches. No GPU-to-host tensor reads or explicit synchronization are added.
- **CI:** the test is discovered by the existing quantization suite and explicitly runs in the B200 kernel job so its SM10x numerics cases can execute. Unsupported hardware skips the GPU cases. B200 coverage has been wired but has not run on this fork branch.
- **Related work / duplicate check (2026-10-20):** [#60891](https://github.com/vllm-project/vllm/pull/60891) and [#60920](https://github.com/vllm-project/vllm/pull/60920) are open and address block-FP8 `wo_a` under forced non-DeepGEMM backends. This change adds an MXFP8 BMM fallback when DeepGEMM is unavailable. Searches for PRs containing `bmm_mxfp8` and open MXFP8 BMM PRs found no duplicate integration.
- **Composition (rechecked 2026-10-20):** checked #60891 at `454e342f38a9f103401729c3e698fd9d71a7d99c` against this refreshed tree. Only an adjacent import hunk conflicts; retaining both imports compiles all four files. Its `maybe_dequant_wo_a` returns early for this kernel's 3D FP8 weight. That PR is not included here.
- **Validation limits:** end-to-end GPU evidence covers automatic selection with DeepGEMM disabled. Explicit `--linear-backend flashinfer_cudnn` overrides and other hardware/attention configurations have no end-to-end validation here. Human review and a model accuracy evaluation remain before upstream submission.
- **AI assistance:** the original patch and draft used Claude Opus 5.5. OpenAI Codex assisted with this branch's preparation, review, CI wiring and PR-body adaptation. This draft does not assert that the human submitter has completed their required review and testing.

---

<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [x] AI assistance was used during the creation of this PR.

- [x] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [ ] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [x] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [x] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>

**BEFORE SUBMITTING, PLEASE READ <https://docs.vllm.ai/en/latest/contributing>** (anything written below this line will be removed by GitHub Actions)
