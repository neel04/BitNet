#include "ggml-bitnet-rsr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ggml-bitnet.h"
#include "ggml.h"
#include "utils.h"

#define QK_I2_S 128
#define QK_I2 128

using namespace std;

const int BLOCK_SIZE = 32;

/**
 * @brief Computes the dot product of `nrc` rows from a 2-bit quantized matrix `vx` and an 8-bit quantized matrix `vy`.
 *
 * @param n      The number of elements per vector (the shared inner dimension K).
 * @param s      Pointer to the destination buffer for the float results.
 * @param bs     Byte stride for the destination buffer `s`.
 * @param vx     Pointer to the 2-bit quantized matrix (e.g., weights).
 * @param bx     Byte stride for the `vx` matrix.
 * @param vy     Pointer to the 8-bit quantized matrix (e.g., activations).
 * @param by     Byte stride for the `vy` matrix.
 * @param nrc    The number of row dot products to compute.
 */
void ggml_rsr_vec_dot_i2_i8_s(
    int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc) {
    const int K = static_cast<int>(ceil(log2(n) - log2(log2(n)))); // Usually 8

    const uint8_t *x = (uint8_t *)vx;
    const int8_t *y = (int8_t *)vy;

    const int nb = (n * 4) / QK_I2_S;             // number of blocks
    const int group_bnum = nb / 32;               // number of 32-groups
    const int la_num = nb % 32;                   // number of leftovers
    const int groupla_num = nb % 32 != 0 ? 1 : 0; // 1 or 0 groups to bunch all leftover elems in

    // Load a tile and unpack out the 2x binary vectors
    for (int group = 0; group < group_bnum; group++) {
        const uint8_t *group_start = x + group * BLOCK_SIZE * BLOCK_SIZE;
        const int8_t *vec_start = y + group * BLOCK_SIZE * BLOCK_SIZE;

        // Initialize buffers with 4x cols
        vector<vector<uint8_t>> tile_buffer1(BLOCK_SIZE, vector<uint8_t>(BLOCK_SIZE * 4, 0));
        vector<vector<uint8_t>> tile_buffer2(BLOCK_SIZE, vector<uint8_t>(BLOCK_SIZE * 4, 0));
        vector<vector<int8_t>> vec_buffer(BLOCK_SIZE, vector<int8_t>(BLOCK_SIZE * 4, 0));

        // Loading the weights, processing them, and loading the vectors
        for (uint8_t i = 0; i < BLOCK_SIZE; i++) {
            for (uint8_t j = 0; j < BLOCK_SIZE; j++) {
                uint8_t idx_in_tile = i * BLOCK_SIZE + j;

                // Unpack 32 byte into 32 * 4 = 128 ternary weights
                vector<int8_t> unpacked_ternary = unpack_i2_s(group_start + idx_in_tile, 32);

                // Convert 128 ternary values to 2 binary vectors of 128 elements each
                VecPair<uint8_t> unpacked_bin = ternary_to_binary(unpacked_ternary, 128);

                for (int k = 0; k < 128; k++) {
                    tile_buffer1[i][k] = unpacked_bin.a[k];
                    tile_buffer2[i][k] = unpacked_bin.b[k];
                    vec_buffer[i][k] = vec_start[k];
                }
            }
        }

        // Unpack the permutations and segments
        auto [perms_buf1, segs_buf1] = preprocess(tile_buffer1, K);
        auto [perms_buf2, segs_buf2] = preprocess(tile_buffer2, K);

        vector<vector<int8_t>> seg_sum_1 = seg_sum(vec_buffer[0], perms_buf1, segs_buf1, K);
        vector<vector<int8_t>> bin_k = generateBinaryMatrix(K);
        auto output = rsr_forward(segs_buf1, bin_k, K);
    }

    throw std::runtime_error("RSR Leftover group handling not implemented yet.");

    // Process leftover elements that don't form complete 32x32 blocks
    if (groupla_num > 0) {
        const uint8_t *leftover_start = x + group_bnum * BLOCK_SIZE * BLOCK_SIZE;

        // Initialize buffers with actual leftover dimensions
        vector<vector<uint8_t>> tile_buffer1(la_num, vector<uint8_t>(la_num * 4, 0));
        vector<vector<uint8_t>> tile_buffer2(la_num, vector<uint8_t>(la_num * 4, 0));

        for (uint8_t i = 0; i < la_num; i++) {
            for (uint8_t j = 0; j < la_num; j++) {
                uint8_t idx_in_tile = i * la_num + j;

                __builtin_debugtrap();
                vector<int8_t> unpacked_ternary = unpack_i2_s(leftover_start + idx_in_tile, la_num);
                VecPair<uint8_t> unpacked_bin = ternary_to_binary(unpacked_ternary, la_num);

                for (int k = 0; k < la_num; k++) {
                    __builtin_debugtrap();
                    tile_buffer1[idx_in_tile][k] = unpacked_bin.a[k];
                    tile_buffer2[idx_in_tile][k] = unpacked_bin.b[k];
                }

                __builtin_debugtrap();
                auto processed_buffer1 = preprocess(tile_buffer1, K);
                auto processed_buffer2 = preprocess(tile_buffer2, K);
            }
        }
    }

    throw std::runtime_error("RSR Matmul not implemented yet!");
}

void ggml_bitnet_rsr_mul_mat(const struct ggml_tensor *src0,
                             const struct ggml_tensor *src1,
                             struct ggml_tensor *dst,
                             const int64_t ir0_start,
                             const int64_t ir0_end,
                             const int64_t ir1_start,
                             const int64_t ir1_end,
                             const int64_t num_rows_per_vec_dot,
                             const size_t src1_col_stride,
                             void *wdata,
                             ggml_vec_dot_t vec_dot) {
    print_once("== Using RSR Kernel == ");
    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type vec_dot_type = GGML_TYPE_I8_S;
    const bool src1_cont = ggml_is_contiguous(src1);
    const size_t row_size = ne00 * sizeof(int8_t);

    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    __builtin_debugtrap();

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                const int64_t i13 = (ir1 / (ne12 * ne1));
                const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                const int64_t i03 = i13 / r3;
                const int64_t i02 = i12 / r2;

                const int64_t i1 = i11;
                const int64_t i2 = i12;
                const int64_t i3 = i13;

                const char *src0_row = (const char *)src0->data + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are
                //       using the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char *src1_col =
                    (const char *)wdata
                    + (src1_cont || src1->type != vec_dot_type ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                                                               : (i11 * nb11 + i12 * nb12 + i13 * nb13));

                // hack - bitnet hardcoded assuming packing 4x 2-bits into 1x 8-bit
                const char *src1_col_de = (const char *)wdata + (i11 * nb11 / 4);

                float *dst_col = (float *)((char *)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                // for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //     vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                // }

                float tmp[32];

                const float *scale = (float *)((uint8_t *)(src0->data) + (ne00 * ne01 / 4));
                const float *act_scales = (const float *)((const char *)wdata + (ne11 * ne10));
                const int32_t *act_sums = (const int32_t *)((const char *)act_scales + (ne11) * sizeof(float));

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    if (src0->type == GGML_TYPE_I2_S) {
                        // vec_dot(ne00,
                        //         &tmp[ir0 - iir0],
                        //         (num_rows_per_vec_dot > 1 ? 16 : 0),
                        //         src0_row + ir0 * nb01 / 4,
                        //         (num_rows_per_vec_dot > 1 ? nb01 : 0),
                        //         src1_col_de,
                        //         (num_rows_per_vec_dot > 1 ? src1_col_stride : 0),
                        //         num_rows_per_vec_dot);
                        tmp[ir0 - iir0] = (float) -10.0f;
                        tmp[ir0 - iir0] = (tmp[ir0 - iir0] - act_sums[i1]) / (act_scales[i1]) * (*scale);
                    } else {
                        vec_dot(ne00,
                                &tmp[ir0 - iir0],
                                (num_rows_per_vec_dot > 1 ? 16 : 0),
                                src0_row + ir0 * nb01,
                                (num_rows_per_vec_dot > 1 ? nb01 : 0),
                                src1_col,
                                (num_rows_per_vec_dot > 1 ? src1_col_stride : 0),
                                num_rows_per_vec_dot);
                        tmp[ir0 - iir0] = (float) -10.0f;
                    }
                }

                for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0],
                           tmp + (cn * 16),
                           (std::min(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
                }
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
    const size_t ne00 = src0->ne[0]; // src0 features
    const size_t ne01 = src0->ne[1]; // src0 bsz
    const size_t ne10 = src1->ne[0]; // src1 features
    const size_t ne11 = src1->ne[1]; // src1 bsz

    return 2048;
}
