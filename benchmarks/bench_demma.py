"""Correctness and latency benchmark for the static rank-160 BFP4 DEMMA path."""

import argparse

import torch

from humming.kernel.demma import demma_bfp4_gemm


def make_inputs(m: int, n: int, k: int):
    torch.manual_seed(7)
    device = "cuda"
    code_shape = (n // 128, k // 128, 128, 5, 4)
    magnitudes = torch.randint(-3, 4, (*code_shape, 8), device=device, dtype=torch.int32)
    exponents = torch.randint(-7, 0, code_shape, device=device, dtype=torch.int32)
    selectors = magnitudes.abs() | ((magnitudes < 0).int() << 2)
    shifts = torch.arange(8, device=device, dtype=torch.int64) * 4
    words = (selectors.to(torch.int64) << shifts).sum(-1)
    displaced = (words >> 28) & 7
    relocated = ((displaced & 1) << 3) | (((displaced >> 1) & 1) << 7)
    relocated |= ((displaced >> 2) & 1) << 11
    codes = ((words & 0x07777777) | relocated | ((exponents + 16).to(torch.int64) << 28))
    codes = codes.to(torch.int32)

    y_physical = torch.randint(-3, 4, (n // 128, k // 1024, 128, 160), device=device)
    by = y_physical.to(torch.float8_e4m3fn).transpose(-1, -2)
    a = torch.randn((m, k), device=device, dtype=torch.float16) / 4
    return a, codes, by, magnitudes, exponents


def reference(a, by, magnitudes, exponents):
    m, k = a.shape
    n = magnitudes.shape[0] * 128
    result = torch.zeros((m, n), device=a.device, dtype=torch.float32)
    values = magnitudes.float() * torch.pow(2.0, exponents.float())[..., None]
    x = torch.cat((values[..., :4].reshape(*values.shape[:-2], 16),
                   values[..., 4:].reshape(*values.shape[:-2], 16)), dim=-1)
    x = x.reshape(n // 128, k // 128, 128, 160)
    for n_block in range(n // 128):
        accumulator = torch.zeros((m, 128), device=a.device, dtype=torch.float32)
        for k_block in range(k // 128):
            y = by[n_block, k_block // 8].float()
            weight = torch.zeros((128, 128), device=a.device, dtype=torch.float16)
            for rank_group in range(5):
                rank_slice = slice(rank_group * 32, (rank_group + 1) * 32)
                product = x[n_block, k_block, :, rank_slice] @ y[rank_slice]
                weight = (weight.float() + product).half()
            a_slice = a[:, k_block * 128:(k_block + 1) * 128]
            accumulator += a_slice.float() @ weight.float().T
        result[:, n_block * 128:(n_block + 1) * 128] = accumulator
    return result.half()


def benchmark(kernel, output, repeat: int, graph_batch: int):
    for _ in range(5):
        kernel(output)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_batch):
            kernel(output)
    graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (repeat * graph_batch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--graph-batch", type=int, default=20)
    args = parser.parse_args()
    if args.m <= 0 or args.n % 128 or args.k % 1024:
        parser.error("requires M > 0, N divisible by 128, K divisible by 1024")

    a, codes, by, magnitudes, exponents = make_inputs(args.m, args.n, args.k)
    expected = reference(a, by, magnitudes, exponents)
    for pipelined in (False, True):
        output = torch.empty_like(expected)

        def launch(destination):
            demma_bfp4_gemm(a, codes, by, destination, pipelined=pipelined)

        launch(output)
        torch.cuda.synchronize()
        error = (output.float() - expected.float()).abs()
        max_error = error.max().item()
        close = torch.isclose(output.float(), expected.float(), rtol=0.02, atol=0.02)
        match = close.float().mean().item()
        latency_us = benchmark(launch, output, args.repeat, args.graph_batch)
        print(f"pipelined={pipelined} M={args.m} N={args.n} K={args.k} "
              f"max_error={max_error:.5g} close={match:.4%} latency_us={latency_us:.3f}")
        if match < 1.0:
            mismatch = (~close).nonzero()[:10]
            for row, column in mismatch.tolist():
                print(f"  [{row},{column}] output={output[row,column].item()} "
                      f"expected={expected[row,column].item()}")
            raise AssertionError("DEMMA output differs from reference")


if __name__ == "__main__":
    main()
