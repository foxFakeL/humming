#pragma once

#include <humming/datatype/dequant.cuh>
#include <humming/mma/wgmma.cuh>
#include <humming/utils/ptx/tma.cuh>


// Static ABI-v4 BFP4 TMA tiles. B_y is physically K-major: each logical
// [160, 128] matrix occupies 128 rows of 160 FP8 values.
template <uint32_t kBlockM, class ElementA>
class DemmaTmaLoader {
public:
  static_assert(sizeof(ElementA) == 2);
  static_assert(kBlockM >= 8 && kBlockM <= 64 && kBlockM % 8 == 0);

  static constexpr uint32_t kCodeWords = 128 * 5 * 4;
  static constexpr uint32_t kCodeRows = kCodeWords / 128;
  static constexpr uint32_t kYSlabBytes = 128 * 128;
  static constexpr uint32_t kAStageBytes = kBlockM * 128 * sizeof(ElementA);
  static constexpr uint32_t kLoadBytes = kCodeWords * sizeof(uint32_t) + kAStageBytes;
  static constexpr uint32_t kYLoadBytes = 2 * kYSlabBytes;

  struct alignas(128) Stage {
    alignas(128) uint32_t codes[kCodeWords];
    // The second Y slab contains K=128..159; TMA zero-fills the rest.
    alignas(128) uint8_t by[2][128][128];
    alignas(128) ElementA a[2][kBlockM][64];
  };

private:
  const void *codes_tma;
  const void *by_tma;
  const void *a_tma;

public:
  CUDA_INLINE DemmaTmaLoader(const void *codes_desc, const void *by_desc, const void *a_desc)
      : codes_tma(codes_desc), by_tma(by_desc), a_tma(a_desc) {}

  CUDA_INLINE void prefetch_tma() const {
    prefetch_tensor_map(codes_tma);
    prefetch_tensor_map(by_tma);
    prefetch_tensor_map(a_tma);
  }

  CUDA_INLINE void load(Stage &stage, void *mbar, uint32_t code_block, uint32_t a_row, uint32_t a_k) const {
    tma_load_2d(codes_tma, stage.codes, mbar, 0, code_block * kCodeRows);
    tma_load_2d(a_tma, stage.a[0], mbar, a_k, a_row);
    tma_load_2d(a_tma, stage.a[1], mbar, a_k + 64, a_row);
  }

  CUDA_INLINE void load_by(Stage &stage, void *mbar, uint32_t by_group) const {
    tma_load_2d(by_tma, stage.by[0], mbar, 0, by_group * 128);
    tma_load_2d(by_tma, stage.by[1], mbar, 128, by_group * 128);
  }

  CUDA_INLINE static const uint32_t *stage_codes(const Stage &stage) { return stage.codes; }

  CUDA_INLINE static void make_by_descriptors(const Stage &stage, uint64_t (&descriptors)[5]) {
    PRAGMA_UNROLL
    for (uint32_t rank_group = 0; rank_group < 5; ++rank_group) {
      const uint32_t slab = rank_group / 4;
      const uint32_t byte_offset = (rank_group % 4) * 32;
      const uint8_t *rank_ptr = &stage.by[slab][0][byte_offset];
      descriptors[rank_group] = make_wgmma_smem_desc<128>(cast_smem_ptr_to_uint(rank_ptr));
    }
  }

  CUDA_INLINE static void make_a_descriptors(const Stage &stage, uint64_t (&descriptors)[8]) {
    PRAGMA_UNROLL
    for (uint32_t k_group = 0; k_group < 8; ++k_group) {
      const uint32_t slab = k_group / 4;
      const uint32_t k_offset = (k_group % 4) * 16;
      const ElementA *k_ptr = &stage.a[slab][0][k_offset];
      descriptors[k_group] = make_wgmma_smem_desc<128>(cast_smem_ptr_to_uint(k_ptr));
    }
  }
};
