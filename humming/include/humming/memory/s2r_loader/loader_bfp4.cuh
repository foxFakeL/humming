#pragma once

#include <humming/utils/base.cuh>
#include <humming/utils/ptx/math.cuh>

// ABI-v4 codes are laid out as [128 rows][5 rank groups][4 lane words].
// Each lane word carries eight three-bit sign-magnitude selectors and one
// four-bit exponent code in its most significant nibble.
CUDA_INLINE void load_bfp4_wgmma_fragment(
    const uint32_t *codes, uint32_t rank_group, uint32_t row_offset, uint32_t (&regs_x)[4]) {
  const uint32_t thread_in_warp_group = threadIdx.x & 127;
  const uint32_t lane = thread_in_warp_group & 31;
  const uint32_t row = row_offset + (thread_in_warp_group >> 5) * 16 + (lane >> 2);
  const uint32_t word_in_rank_group = lane & 3;

  // The SM90 WGMMA RS A operand has two rows per thread. Its four uint32_t
  // registers hold [row0 low K, row1 low K, row0 high K, row1 high K].
  const uint32_t word_offset = rank_group * 4 + word_in_rank_group;
  const uint32_t encoded_row0 = codes[row * 20 + word_offset];
  const uint32_t encoded_row1 = codes[(row + 8) * 20 + word_offset];

  const uint32_t displaced_row0 = encoded_row0 & 0x00000888;
  const uint32_t displaced_row1 = encoded_row1 & 0x00000888;
  const uint32_t selectors_row0 =
      (encoded_row0 & 0x07777777) | ((displaced_row0 * 0x02480000) & 0x70000000);
  const uint32_t selectors_row1 =
      (encoded_row1 & 0x07777777) | ((displaced_row1 * 0x02480000) & 0x70000000);

  const uint32_t exponent_row0 = encoded_row0 >> 28;
  const uint32_t exponent_row1 = encoded_row1 >> 28;
  const uint32_t normal_row0 = exponent_row0 * 0x04040400 + 0x01FFFC00;
  const uint32_t normal_row1 = exponent_row1 * 0x04040400 + 0x01FFFC00;
  const uint32_t subnormal_row0 = (exponent_row0 + 1) * 0x03020100;
  const uint32_t subnormal_row1 = (exponent_row1 + 1) * 0x03020100;
  const uint32_t subnormal_mask_row0 = 0u - (exponent_row0 < 2);
  const uint32_t subnormal_mask_row1 = 0u - (exponent_row1 < 2);
  const uint32_t positive_lut_row0 =
      normal_row0 ^ ((normal_row0 ^ subnormal_row0) & subnormal_mask_row0);
  const uint32_t positive_lut_row1 =
      normal_row1 ^ ((normal_row1 ^ subnormal_row1) & subnormal_mask_row1);

  regs_x[0] = prmt(positive_lut_row0, positive_lut_row0 | 0x80808080, selectors_row0);
  regs_x[1] = prmt(positive_lut_row1, positive_lut_row1 | 0x80808080, selectors_row1);
  regs_x[2] = prmt(positive_lut_row0, positive_lut_row0 | 0x80808080, selectors_row0 >> 16);
  regs_x[3] = prmt(positive_lut_row1, positive_lut_row1 | 0x80808080, selectors_row1 >> 16);
}
