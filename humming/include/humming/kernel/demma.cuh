#pragma once

#include <cuda_fp16.h>
#include <humming/memory/g2s_pipeline_demma.cuh>
#include <humming/mma/demma.cuh>

// Static ABI-v4 BFP4 GEMM: C[M, N] = A[M, K] @ decode(codes, Y)[N, K].T.
// The launch has one TMA producer warpgroup and two WGMMA consumer warpgroups.
template <class Fp8MmaOpClass, class F16MmaOpClass, uint32_t kBlockM, bool kPipelined>
__global__ __launch_bounds__(384) void humming_demma(
    const __grid_constant__ CUtensorMap codes_map,
    const __grid_constant__ CUtensorMap by_map,
    const __grid_constant__ CUtensorMap a_map,
    half *output, uint32_t shape_m, uint32_t shape_n, uint32_t shape_k) {
  using Pipeline = DemmaTmaPipeline<kBlockM, half>;
  using Mma = DEMMA<Fp8MmaOpClass, F16MmaOpClass>;
  extern __shared__ uint4 shared_memory[];
  auto &storage = *reinterpret_cast<typename Pipeline::SharedStorage *>(shared_memory);
  Pipeline pipeline(storage, &codes_map, &by_map, &a_map);
  pipeline.init();

  const uint32_t n_block = blockIdx.x;
  const uint32_t m_block = blockIdx.y;
  const uint32_t k_blocks = shape_k / 128;
  const uint32_t k_blocks_per_n = shape_k / 128;
  const uint32_t by_groups_per_n = shape_k / 1024;

  if (threadIdx.x == 0) {
    pipeline.prefetch_tma();
    for (uint32_t iteration = 0; iteration < k_blocks; ++iteration) {
      if (iteration % 8 == 0) {
        const uint32_t by_group = n_block * by_groups_per_n + iteration / 8;
        pipeline.load_by(iteration / 8, by_group);
      }
      const uint32_t code_block = n_block * k_blocks_per_n + iteration;
      pipeline.load(iteration, code_block, m_block * kBlockM, iteration * 128);
    }
  }

  if (threadIdx.x >= 128) {
    const uint32_t consumer_id = (threadIdx.x - 128) / 128;
    const uint32_t row_offset = consumer_id * 64;
    Mma mma;
    mma.zero_accum();

    if constexpr (kPipelined) {
      pipeline.wait(0);
      pipeline.wait_by(0);
      uint64_t by_descriptors[5];
      pipeline.make_by_descriptors(0, by_descriptors);
      mma.prime(pipeline.stage_codes(0), by_descriptors, row_offset);
      mma.wait_all();

      for (uint32_t iteration = 0; iteration < k_blocks; ++iteration) {
        const bool has_next = iteration + 1 < k_blocks;
        uint64_t a_descriptors[8];
        pipeline.make_a_descriptors(iteration, a_descriptors);
        if (has_next) {
          pipeline.wait(iteration + 1);
          pipeline.wait_by((iteration + 1) / 8);
          uint64_t next_by_descriptors[5];
          pipeline.make_by_descriptors((iteration + 1) / 8, next_by_descriptors);
          mma.issue_next_fp8(pipeline.stage_codes(iteration + 1), next_by_descriptors, row_offset);
        }
        mma.begin_f16(a_descriptors);
        mma.wait_all();
        if ((threadIdx.x & 127) == 0) {
          pipeline.release(iteration);
          if (iteration % 8 == 7) pipeline.release_by(iteration / 8);
        }
        if (has_next) mma.advance_buffer();
      }
    } else {
      for (uint32_t iteration = 0; iteration < k_blocks; ++iteration) {
        pipeline.wait(iteration);
        pipeline.wait_by(iteration / 8);
        uint64_t by_descriptors[5];
        uint64_t a_descriptors[8];
        pipeline.make_by_descriptors(iteration / 8, by_descriptors);
        pipeline.make_a_descriptors(iteration, a_descriptors);
        mma.run(pipeline.stage_codes(iteration), by_descriptors, a_descriptors, row_offset);
        if ((threadIdx.x & 127) == 0) {
          pipeline.release(iteration);
          if (iteration % 8 == 7) pipeline.release_by(iteration / 8);
        }
      }
    }

    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warp = (threadIdx.x & 127) >> 5;
    const uint32_t row = n_block * 128 + row_offset + warp * 16 + (lane >> 2);
    const uint32_t column = m_block * kBlockM + (lane & 3) * 2;
    const float *accumulator = mma.template final_regs_c_as_ptr<float>();
    PRAGMA_UNROLL
    for (uint32_t column_pair = 0; column_pair < kBlockM / 8; ++column_pair) {
      const uint32_t output_column = column + column_pair * 8;
      const uint32_t reg = column_pair * 4;
      if (output_column < shape_m) {
        output[output_column * shape_n + row] = __float2half(accumulator[reg]);
        output[output_column * shape_n + row + 8] = __float2half(accumulator[reg + 2]);
      }
      if (output_column + 1 < shape_m) {
        output[(output_column + 1) * shape_n + row] = __float2half(accumulator[reg + 1]);
        output[(output_column + 1) * shape_n + row + 8] = __float2half(accumulator[reg + 3]);
      }
    }
  }
}
