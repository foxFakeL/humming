#pragma once

#include <cuda_awbarrier_primitives.h>
#include <humming/memory/g2s_loader/loader_demma.cuh>
#include <humming/utils/ptx/barrier.cuh>
#include <humming/utils/ptx/tma.cuh>


// Two-stage producer/consumer synchronization for static ABI-v4 BFP4 TMA tiles.
template <uint32_t kBlockM, class ElementA>
class DemmaTmaPipeline {
public:
  using Loader = DemmaTmaLoader<kBlockM, ElementA>;
  using Stage = typename Loader::Stage;

  static constexpr uint32_t kStages = 2;
  static constexpr uint32_t kConsumerWarpGroups = 2;
  static constexpr uint32_t kLoadBytes = Loader::kLoadBytes;
  static constexpr uint32_t kYLoadBytes = Loader::kYLoadBytes;

  struct SharedStorage {
    alignas(16) uint64_t load_mbar[kStages];
    alignas(16) uint64_t load_empty_mbar[kStages];
    alignas(16) uint64_t by_mbar[kStages];
    alignas(16) uint64_t by_empty_mbar[kStages];
    alignas(128) Stage stages[kStages];
  };

  SharedStorage &smem;
  Loader loader;

  CUDA_INLINE DemmaTmaPipeline(
      SharedStorage &storage, const void *codes_desc,
      const void *by_desc, const void *a_desc)
      : smem(storage), loader(codes_desc, by_desc, a_desc) {
  }

  CUDA_INLINE void init() {
    if (threadIdx.x == 0) {
      for (uint32_t stage = 0; stage < kStages; ++stage) {
        __mbarrier_init(&smem.load_mbar[stage], 1);
        __mbarrier_init(&smem.load_empty_mbar[stage], kConsumerWarpGroups);
        __mbarrier_init(&smem.by_mbar[stage], 1);
        __mbarrier_init(&smem.by_empty_mbar[stage], kConsumerWarpGroups);
      }
    }
    mbarrier_init_sync<false>();
  }

  CUDA_INLINE void prefetch_tma() const {
    if (threadIdx.x == 0) {
      loader.prefetch_tma();
    }
  }

  // One producer lane calls this once per K=128 block. Its consumer must
  // release the stage after all WGMMA reads have completed.
  CUDA_INLINE void load(uint32_t iteration, uint32_t code_block,
                        uint32_t a_row, uint32_t a_k) {
    const uint32_t stage = iteration % kStages;
    if (iteration >= kStages) {
      const uint32_t previous_phase = (iteration / kStages - 1) & 1;
      mbarrier_wait(&smem.load_empty_mbar[stage], previous_phase);
    }

    tma_expect_tx(&smem.load_mbar[stage], kLoadBytes);
    loader.load(smem.stages[stage], &smem.load_mbar[stage], code_block, a_row, a_k);
  }

  // B_y is reused by eight consecutive K=128 blocks.
  CUDA_INLINE void load_by(uint32_t group_iteration, uint32_t by_group) {
    const uint32_t stage = group_iteration % kStages;
    if (group_iteration >= kStages) {
      const uint32_t previous_phase = (group_iteration / kStages - 1) & 1;
      mbarrier_wait(&smem.by_empty_mbar[stage], previous_phase);
    }

    tma_expect_tx(&smem.by_mbar[stage], kYLoadBytes);
    loader.load_by(smem.stages[stage], &smem.by_mbar[stage], by_group);
  }

  CUDA_INLINE void wait(uint32_t iteration) const {
    const uint32_t stage = iteration % kStages;
    const uint32_t phase = (iteration / kStages) & 1;
    mbarrier_wait(&smem.load_mbar[stage], phase);
  }

  CUDA_INLINE void wait_by(uint32_t group_iteration) const {
    const uint32_t stage = group_iteration % kStages;
    const uint32_t phase = (group_iteration / kStages) & 1;
    mbarrier_wait(&smem.by_mbar[stage], phase);
  }

  // One leader from each consumer warpgroup calls release after that group's
  // shared reads and WGMMA operations have completed.
  CUDA_INLINE void release(uint32_t iteration) {
    mbarrier_arrive(&smem.load_empty_mbar[iteration % kStages]);
  }

  CUDA_INLINE void release_by(uint32_t group_iteration) {
    mbarrier_arrive(&smem.by_empty_mbar[group_iteration % kStages]);
  }

  CUDA_INLINE const uint32_t *stage_codes(uint32_t iteration) const {
    return Loader::stage_codes(smem.stages[iteration % kStages]);
  }

  CUDA_INLINE void make_by_descriptors(
      uint32_t group_iteration, uint64_t (&descriptors)[5]) const {
    Loader::make_by_descriptors(smem.stages[group_iteration % kStages], descriptors);
  }

  CUDA_INLINE void make_a_descriptors(
      uint32_t iteration, uint64_t (&descriptors)[8]) const {
    Loader::make_a_descriptors(smem.stages[iteration % kStages], descriptors);
  }
};
