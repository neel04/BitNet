#include "ggml-bitnet-rsr.h"

#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "ggml.h"
#include "utils.h"

#define QK_I2_S 128
#define QK_I2 128

using namespace std;

void ggml_bitnet_rsr_mul_mat(const struct ggml_tensor *src0,
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
                             const ggml_vec_dot_t vec_dot) {
    (void)num_rows_per_vec_dot;
    (void)src1_col_stride;

    print_once("== Using RSR Kernel ==");

    assert(num_rows_per_vec_dot == 1); // Simplify things

    GGML_TENSOR_BINARY_OP_LOCALS

    // Get scales and sums
    const float *scale = (float *)((uint8_t *)(src0->data) + (ne00 * ne01 / 4));
    const float *act_scales = (const float *)((const char *)wdata + (ne11 * ne10));
    const int32_t *act_sums = (const int32_t *)((const char *)act_scales + (ne11) * sizeof(float));

    const bool src1_cont = ggml_is_contiguous(src1);

    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    int blck_0, blck_1;
    blck_0 = blck_1 = 16; // FIX: Hardcoded chunk size

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = ir1_start; ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                // Calculate indices
                const int64_t i13 = (ir1 / (ne12 * ne1));
                const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // Broadcast indices
                const int64_t i03 = i13 / r3;
                const int64_t i02 = i12 / r2;

                const int64_t i1 = i11;
                const int64_t i2 = i12;
                const int64_t i3 = i13;

                // Get weight base pointer with broadcasting
                const uint8_t *src0_row = (const uint8_t *)src0->data + (i02 * nb02 + i03 * nb03);

                // Get activation vector - nb11/4 because int8 is 1/4 size of float32
                const int8_t *src1_col_de = (const int8_t *)wdata + (i11 * nb11 / 4);

                // Get output pointer
                float *dst_col = (float *)((char *)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));
                const char *src1_col =
                    (const char *)wdata
                    + (src1_cont || src1->type != vec_dot_type ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                                                               : (i11 * nb11 + i12 * nb12 + i13 * nb13));

                // Number of output rows to compute in this chunk
                const int output_rows = std::min(blck_0, (int)(ir0_end - iir0));

                float tmp[32];

                if (src0->type == GGML_TYPE_I2_S) {
                    vector<vector<int8_t>> weight_matrix(output_rows, vector<int8_t>(ne00, 0));
                    vector<int8_t> acts(ne00, 0);

                    // Load activations
                    for (int64_t i = 0; i < ne00; ++i) {
                        acts[i] = src1_col_de[i];
                    }

                    // Unpack weights row by row using the fixed unpack_i2_s
                    for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0++) {
                        const uint8_t *packed_row = src0_row + (ir0 * nb01 / 4);

                        vector<int8_t> unpacked_row = unpack_i2_s(packed_row, ne00 / 4);

                        // Copy to weight matrix (only take ne00 weights in case of padding)
                        for (int64_t j = 0; j < ne00 && j < unpacked_row.size(); j++) {
                            weight_matrix[ir0 - iir0][j] = unpacked_row[j];
                        }
                    }

                    vector<float> output = vectorMatrixMultiply(acts, weight_matrix);

                    for (int idx = 0; idx < output_rows; idx++) {
                        float result = output[idx];
                        // Transforming {0, 1, 2} ==> {-1, 0, q}
                        result = (result - act_sums[i1]) / act_scales[i1] * (*scale);
                        tmp[idx] = result;
                    }
                } else {
                    // Fallback to default GGML Kernel to handle higher precision gemv
                    for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                        vec_dot(ne00,
                                &tmp[ir0 - iir0],
                                (num_rows_per_vec_dot > 1 ? 16 : 0),
                                src0_row + ir0 * nb01,
                                (num_rows_per_vec_dot > 1 ? nb01 : 0),
                                src1_col,
                                (num_rows_per_vec_dot > 1 ? src1_col_stride : 0),
                                num_rows_per_vec_dot);
                    }
                }

                memcpy(&dst_col[iir0], tmp, output_rows * sizeof(float));
            }
        }
    }
}

bool ggml_bitnet_rsr_can_mul_mat(const struct ggml_tensor *src0,
                                 const struct ggml_tensor *src1,
                                 const struct ggml_tensor *dst) {
    // For now, support the same conditions as regular BitNet
    return src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && src0->backend == GGML_BACKEND_TYPE_CPU;
}
size_t ggml_bitnet_rsr_mul_mat_get_wsize(const struct ggml_tensor *src0,
                                         const struct ggml_tensor *src1,
                                         const struct ggml_tensor *dst) {
    (void)dst; // Silence unused parameter warning

    const size_t ne00 = src0->ne[0]; // src0 features (columns)
    const size_t ne01 = src0->ne[1]; // src0 rows
    const size_t ne10 = src1->ne[0]; // src1 features
    const size_t ne11 = src1->ne[1]; // src1 batch size
    (void)ne10;
    (void)ne11; // Silence unused variable warnings

    // Workspace needed for RSR preprocessing and computation
    // This includes space for permutations, segmentations, and intermediate results
    const int K = static_cast<int>(ceil(log2(ne00) - log2(log2(ne00))));

    // Estimate: permutations + segmentations + temporary buffers
    // Each permutation/segmentation array: (ne00/K) * max(ne00, pow(2,K)) * sizeof(int8_t)
    // Plus buffer for results and intermediate computations
    size_t workspace_size = 4 * (ne00 / K + 1) * std::max(ne00, (size_t)pow(2, K)) * sizeof(int8_t);
    workspace_size += ne01 * sizeof(float); // For result buffer

    return workspace_size;
}
