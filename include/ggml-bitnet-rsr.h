#ifndef GGML_BITNET_RSR_H
#define GGML_BITNET_RSR_H

#include <stddef.h>
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

GGML_API void
ggml_rsr_vec_dot_i2_i8_s(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);

GGML_API void ggml_bitnet_rsr_mul_mat(const struct ggml_tensor *src0,
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

GGML_API bool ggml_bitnet_rsr_can_mul_mat(const struct ggml_tensor *src0,
                                          const struct ggml_tensor *src1,
                                          const struct ggml_tensor *dst);

GGML_API size_t ggml_bitnet_rsr_mul_mat_get_wsize(const struct ggml_tensor *src0,
                                                  const struct ggml_tensor *src1,
                                                  const struct ggml_tensor *dst);

#ifdef __cplusplus
}
#endif

#endif // GGML_BITNET_RSR_H
