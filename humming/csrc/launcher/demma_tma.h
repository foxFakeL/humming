#pragma once

#include <cstdint>
#include <cuda.h>


struct DemmaTmaMaps {
  CUtensorMap codes;
  CUtensorMap by;
  CUtensorMap a;
};

inline CUresult make_demma_tma_map_2d(
    CUtensorMap *tensor_map, const void *base, CUtensorMapDataType dtype,
    uint64_t width, uint64_t rows, uint64_t row_stride_bytes,
    uint32_t tile_width, uint32_t tile_rows,
    CUtensorMapSwizzle swizzle) {
  const uint64_t global_shape[2] = {width, rows};
  const uint64_t global_strides[1] = {row_stride_bytes};
  const uint32_t box_shape[2] = {tile_width, tile_rows};
  const uint32_t element_strides[2] = {1, 1};
  return cuTensorMapEncodeTiled(
      tensor_map, dtype, 2, const_cast<void *>(base), global_shape,
      global_strides, box_shape, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// codes: [N/128, K/128, 128, 5, 4] uint32, contiguous.
// by: [N/128, K/1024, 160, 128] FP8, with the final two logical modes
// physically transposed so each group is [128, 160] contiguous.
// a: [M, K] float16, contiguous.
inline CUresult make_demma_tma_maps(
    DemmaTmaMaps *maps, const void *codes, const void *by, const void *a,
    uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
    uint32_t block_m, CUtensorMapDataType a_dtype) {
  if (shape_m == 0 || shape_n == 0 || shape_k == 0 ||
      shape_n % 128 != 0 || shape_k % 1024 != 0 ||
      block_m < 8 || block_m > 64 || block_m % 8 != 0 ||
      a_dtype != CU_TENSOR_MAP_DATA_TYPE_FLOAT16) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  const uint64_t code_blocks = uint64_t(shape_n / 128) * (shape_k / 128);
  const uint64_t by_groups = uint64_t(shape_n / 128) * (shape_k / 1024);

  CUresult result = make_demma_tma_map_2d(
      &maps->codes, codes, CU_TENSOR_MAP_DATA_TYPE_UINT32,
      128, code_blocks * 20, 128 * sizeof(uint32_t),
      128, 20, CU_TENSOR_MAP_SWIZZLE_NONE);
  if (result != CUDA_SUCCESS) return result;

  result = make_demma_tma_map_2d(
      &maps->by, by, CU_TENSOR_MAP_DATA_TYPE_UINT8,
      160, by_groups * 128, 160,
      128, 128, CU_TENSOR_MAP_SWIZZLE_128B);
  if (result != CUDA_SUCCESS) return result;

  return make_demma_tma_map_2d(
      &maps->a, a, a_dtype,
      shape_k, shape_m, uint64_t(shape_k) * 2,
      64, block_m, CU_TENSOR_MAP_SWIZZLE_128B);
}
