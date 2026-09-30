import dataclasses

import torch

from humming.config.mma import WgmmaOpClassImpl
from humming.jit.runtime import KernelRuntime
from humming.ops.utils import init_humming_launcher


@dataclasses.dataclass(kw_only=True)
class DemmaKernel(KernelRuntime):
    block_m: int
    pipelined: bool = True
    by_dtype: torch.dtype = torch.float8_e4m3fn

    def __post_init__(self):
        KernelRuntime.__post_init__(self)

    def init_kernel(self):
        if self.sm_version != 90:
            raise ValueError("DEMMA requires Hopper SM90")
        if self.block_m not in range(8, 65, 8):
            raise ValueError("block_m must be a multiple of eight in [8, 64]")
        if self.by_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise ValueError("Y must be float8_e4m3fn or float8_e5m2")

        by_ptx_dtype = "e4m3" if self.by_dtype == torch.float8_e4m3fn else "e5m2"
        fp8_op = WgmmaOpClassImpl(128, 64, 32, by_ptx_dtype, "e5m2", "f16")
        f16_op = WgmmaOpClassImpl(self.block_m, 64, 16, "f16", "f16", "f32")
        pipeline_literal = "true" if self.pipelined else "false"
        self.kernel_expr = (
            f"humming_demma<Fp8MmaOpClass, F16MmaOpClass, {self.block_m}, "
            f"{pipeline_literal}>"
        )
        fp8_definition = fp8_op.to_cpp_str(include_class_name=True)
        f16_definition = f16_op.to_cpp_str(include_class_name=True)
        self.code = "\n".join(
            [
                "#include <humming/kernel/demma.cuh>",
                fp8_definition.replace("class MmaOpClass", "struct Fp8MmaOpClass"),
                f16_definition.replace("class MmaOpClass", "struct F16MmaOpClass"),
                f"using Storage = DemmaTmaPipeline<{self.block_m}, half>::SharedStorage;",
                'extern "C" __constant__ uint32_t SMEM_SIZE = sizeof(Storage);',
                'extern "C" __constant__ uint32_t NUM_THREADS = 384;',
                f'extern "C" __constant__ uint32_t BLOCK_M = {self.block_m};',
            ]
        )
        self.prepare()
        init_humming_launcher()
        self.kernel_id, self.kernel_name = torch.ops.humming.register_demma_kernel(self.kernel_filename)

    def __call__(
        self,
        a: torch.Tensor,
        codes: torch.Tensor,
        by: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output is None:
            output = torch.empty((a.shape[0], codes.shape[0] * 128), dtype=a.dtype, device=a.device)
        if codes.dtype == torch.uint32:
            codes = codes.view(torch.int32)
        if by.dtype != self.by_dtype:
            raise ValueError(f"Y dtype {by.dtype} does not match compiled dtype {self.by_dtype}")
        by = by.view(torch.uint8)
        torch.ops.humming.launch_demma(self.kernel_id, a, codes, by, output, self.block_m)
        return output


def demma_bfp4_gemm(
    a: torch.Tensor,
    codes: torch.Tensor,
    by: torch.Tensor,
    output: torch.Tensor | None = None,
    *,
    pipelined: bool = True,
    block_m: int | None = None,
) -> torch.Tensor:
    """Compute ``A @ W.T`` from ABI-v4 codes and rank-160 FP8 factors.

    ``codes`` has shape ``[N/128, K/128, 128, 5, 4]``. ``by`` has logical
    shape ``[N/128, K/1024, 160, 128]`` with the final modes stored K-major.
    The float16 output has shape ``[M, N]``.
    """
    if block_m is None:
        block_m = min(64, max(8, (a.shape[0] + 7) // 8 * 8))
    return DemmaKernel(block_m=block_m, pipelined=pipelined, by_dtype=by.dtype)(a, codes, by, output)
