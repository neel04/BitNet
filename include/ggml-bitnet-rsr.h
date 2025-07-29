#ifndef GGML_BITNET_RSR_H
#define GGML_BITNET_RSR_H

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

void ggml_rsr_vec_dot_i2_i8_s(
    int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);

void ggml_bitnet_rsr_mul_mat(struct ggml_tensor *dst, const struct ggml_tensor *src0, const struct ggml_tensor *src1);

bool ggml_bitnet_rsr_can_mul_mat(const struct ggml_tensor *src0,
                                 const struct ggml_tensor *src1,
                                 const struct ggml_tensor *dst);

size_t ggml_bitnet_rsr_mul_mat_get_wsize(const struct ggml_tensor *src0,
                                         const struct ggml_tensor *src1,
                                         const struct ggml_tensor *dst);

#ifdef __cplusplus
}
#endif

#endif // GGML_BITNET_RSR_H
