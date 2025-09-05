#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// FastBin (Fast Binary) kernel functions
GGML_API void ggml_bitnet_fastbin_mul_mat(const struct ggml_tensor *src0,
                                          const struct ggml_tensor *src1,
                                          struct ggml_tensor *dst,
                                          const int64_t ir0_start,
                                          const int64_t ir0_end,
                                          const int64_t ir1_start,
                                          const int64_t ir1_end,
                                          const int64_t num_rows_per_vec_dot,
                                          const size_t row_size,
                                          const size_t src1_col_stride,
                                          void *wdata,
                                          const enum ggml_type vec_dot_type,
                                          const ggml_vec_dot_t vec_dot);

#ifdef __cplusplus
}
#endif
