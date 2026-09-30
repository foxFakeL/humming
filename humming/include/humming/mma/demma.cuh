#pragma once

#include <cuda_fp16.h>
#include <humming/memory/s2r_loader/loader_bfp4.cuh>
#include <humming/utils/base.cuh>
#include <humming/utils/ptx/wgmma.cuh>

#include <cstdint>
#include <type_traits>

// Static-rank (160) BFP4 decode followed by the two Hopper WGMMA operations:
//   half(X_e5m2 @ Y_fp8) @ A_half.T -> C_float
// One consumer warpgroup owns 64 X rows. The producer supplies descriptors for
// five rank-32 slices of Y and eight K-16 slices of A from the current stages.
// Both operation classes use the project's swapped WGMMA convention: their
// register operand is the PTX A operand and their descriptor is the PTX B operand.
template <class Fp8MmaOpClass, class F16MmaOpClass>
struct DEMMA {
  using Fp8MmaShape = typename Fp8MmaOpClass::MmaShape;
  using F16MmaShape = typename F16MmaOpClass::MmaShape;
  using IntermediateRegisters = typename Fp8MmaOpClass::CRegisters;
  using CRegistersArrayType = typename F16MmaOpClass::CRegisters;
  using CRegister = float;

  static constexpr uint32_t kRankGroups = 5;
  static constexpr uint32_t kOutputKGroups = 8;

  static_assert(Fp8MmaShape::M == 128 && Fp8MmaShape::N == 64 && Fp8MmaShape::K == 32);
  static_assert(F16MmaShape::N == 64 && F16MmaShape::K == 16);
  static_assert(F16MmaShape::M >= 8 && F16MmaShape::M <= 64 && F16MmaShape::M % 8 == 0);
  static_assert(Fp8MmaOpClass::kATypeBits == 8 && Fp8MmaOpClass::kBTypeBits == 8);
  static_assert(Fp8MmaOpClass::kCTypeBits == 16);
  static_assert(std::is_same<typename Fp8MmaOpClass::ValTypeC, half>::value);
  static_assert(F16MmaOpClass::kATypeBits == 16 && F16MmaOpClass::kBTypeBits == 16);
  static_assert(F16MmaOpClass::kCTypeBits == 32);
  static_assert(std::is_same<typename F16MmaOpClass::ValTypeC, float>::value);
  static_assert(std::is_same<IntermediateRegisters, uint32_t[32]>::value);
  static_assert(std::is_same<CRegistersArrayType, float[F16MmaShape::M / 2]>::value);
  static_assert(sizeof(typename Fp8MmaOpClass::BRegisters) == 4 * sizeof(uint32_t));
  static_assert(sizeof(typename F16MmaOpClass::BRegisters) == 4 * sizeof(uint32_t));
  static_assert(sizeof(IntermediateRegisters) == kOutputKGroups * sizeof(typename F16MmaOpClass::BRegisters));

  alignas(16) IntermediateRegisters regs_b[2];
  uint32_t decoded_x[kRankGroups][4];
  alignas(16) CRegistersArrayType regs_c;
  uint32_t current_buffer = 0;

  CUDA_INLINE void zero_accum() {
    current_buffer = 0;
    CRegister *accumulator = regs_c_as_ptr<CRegister>();
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < sizeof(regs_c) / sizeof(CRegister); i++) {
      accumulator[i] = 0;
    }
  }

  CUDA_INLINE void run(
      const uint32_t *codes_stage,
      const uint64_t (&by_descriptors)[kRankGroups],
      const uint64_t (&a_descriptors)[kOutputKGroups],
      uint32_t row_offset) {
    prime(codes_stage, by_descriptors, row_offset);
    wait_all();
    begin_f16(a_descriptors);
    wait_all();
  }

  // Prepare the first K=128 block before entering the two-level register
  // pipeline. The caller waits before FP16 WGMMA consumes this result.
  CUDA_INLINE void prime(
      const uint32_t *codes_stage,
      const uint64_t (&by_descriptors)[kRankGroups],
      uint32_t row_offset) {
    current_buffer = 0;
    issue_fp8<0>(codes_stage, by_descriptors, row_offset);
  }

  // The main loop calls this after issuing FP8 for the next block and FP16
  // for the current block, before consuming results or releasing stages.
  CUDA_INLINE void wait_all() {
    wgmma_wait<0>();
    // Keep decoded fragments live through the wait so the compiler cannot
    // reuse them while an asynchronous FP8 WGMMA still reads them.
    PRAGMA_UNROLL
    for (uint32_t rank_group = 0; rank_group < kRankGroups; ++rank_group) {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < 4; ++i) {
        warpgroup_fence_operand(decoded_x[rank_group][i]);
      }
    }
  }

  // The current FP8 result was completed by the previous stage's wait_all.
  // This FP16 WGMMA reads the current buffer while FP8 writes the other one.
  CUDA_INLINE void begin_f16(const uint64_t (&a_descriptors)[kOutputKGroups]) {
    if (current_buffer == 0) {
      issue_f16<0>(a_descriptors);
    } else {
      issue_f16<1>(a_descriptors);
    }
  }

  // Issue next block's FP8 WGMMA before current block's FP16 WGMMA.
  CUDA_INLINE void issue_next_fp8(
      const uint32_t *codes_next,
      const uint64_t (&by_descriptors_next)[kRankGroups],
      uint32_t row_offset) {
    if (current_buffer == 0) {
      issue_fp8<1>(codes_next, by_descriptors_next, row_offset);
    } else {
      issue_fp8<0>(codes_next, by_descriptors_next, row_offset);
    }
  }

  CUDA_INLINE void advance_buffer() { current_buffer ^= 1; }

private:
  template <uint32_t kBuffer>
  CUDA_INLINE void issue_fp8(
      const uint32_t *codes_stage,
      const uint64_t (&by_descriptors)[kRankGroups],
      uint32_t row_offset) {
    uint32_t *intermediate = reinterpret_cast<uint32_t *>(&regs_b[kBuffer]);

    PRAGMA_UNROLL
    for (uint32_t rank_group = 0; rank_group < kRankGroups; rank_group++) {
      auto &fragment = decoded_x[rank_group];
      load_bfp4_wgmma_fragment(codes_stage, rank_group, row_offset, fragment);

      // The decoded fragment is the register A operand of the FP8 WGMMA.
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < 4; i++) {
        warpgroup_fence_operand(fragment[i]);
      }
      wgmma_fence();
      uint64_t by_descriptor = by_descriptors[rank_group];
      Fp8MmaOpClass::fma(by_descriptor, fragment, intermediate, rank_group != 0);
    }
    wgmma_commit();
  }

  template <uint32_t kBuffer>
  CUDA_INLINE void issue_f16(const uint64_t (&a_descriptors)[kOutputKGroups]) {
    uint32_t *intermediate = reinterpret_cast<uint32_t *>(&regs_b[kBuffer]);
    // The FP16 accumulator has the same per-thread register layout as eight
    // consecutive K-16 register A fragments of the second WGMMA.
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < sizeof(regs_b[kBuffer]) / sizeof(uint32_t); i++) {
      warpgroup_fence_operand(intermediate[i]);
    }
    wgmma_fence();
    PRAGMA_UNROLL
    for (uint32_t k_group = 0; k_group < kOutputKGroups; k_group++) {
      uint64_t a_descriptor = a_descriptors[k_group];
      uint32_t *b_fragment = intermediate + k_group * 4;
      F16MmaOpClass::fma(a_descriptor, b_fragment, regs_c_as_ptr<CRegister>(), true);
    }
    wgmma_commit();
  }

public:
  template <class T = uint32_t>
  CUDA_INLINE T *regs_c_as_ptr() {
    return reinterpret_cast<T *>(&regs_c);
  }

  template <class T = uint32_t>
  CUDA_INLINE T *final_regs_c_as_ptr() {
    return regs_c_as_ptr<T>();
  }

  static constexpr uint32_t final_regs_c_index() { return 0; }
};
