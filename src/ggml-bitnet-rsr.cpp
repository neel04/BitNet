#include "ggml-bitnet-rsr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cassert>
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

const int BLOCK_SIZE = 16;
const int TILE_SIZE = 8;
const int K = 8;

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
    const uint8_t *x = (uint8_t *)vx;
    const int8_t *y = (int8_t *)vy;

    const int nb = n / QK_I2_S;                           // number of blocks
    const int group_bnum = nb / BLOCK_SIZE;               // number of BLOCK_SIZE-groups
    const int la_num = nb % BLOCK_SIZE;                   // number of leftovers
    const int groupla_num = nb % BLOCK_SIZE != 0 ? 1 : 0; // 1 or 0 groups to bunch all leftover elems in

    // Load a tile and unpack out the 2x binary vectors
    for (int group = 0; group < group_bnum; group++) {
        const uint8_t *group_start = x + group * BLOCK_SIZE * BLOCK_SIZE;

        // Initialize buffers with 4x rows
        vector<vector<uint8_t>> tile_buffer1(TILE_SIZE * 4, vector<uint8_t>(TILE_SIZE, 0));
        vector<vector<uint8_t>> tile_buffer2(TILE_SIZE * 4, vector<uint8_t>(TILE_SIZE, 0));

        for (uint8_t i = 0; i < TILE_SIZE; i++) {
            for (uint8_t j = 0; j < TILE_SIZE; j++) {
                uint8_t idx_in_tile = i * TILE_SIZE + j;

                // Unpack 1 byte into 4 ternary values
                vector<int8_t> unpacked_ternary = unpack_i2_s(group_start + idx_in_tile, 1);

                // Convert 4 ternary values to 2 binary vectors of 4 elements each
                VecPair<uint8_t> unpacked_bin = ternary_to_binary(unpacked_ternary, 4);

                for (int k = 0; k < 4; k++) {
                    tile_buffer1[i * 4 + k][j] = unpacked_bin.a[k];
                    tile_buffer2[i * 4 + k][j] = unpacked_bin.b[k];
                }

                __builtin_debugtrap();
                auto processed_buffer1 = preprocess(tile_buffer1, K);
                auto processed_buffer2 = preprocess(tile_buffer2, K);
                __builtin_debugtrap();
            }
        }

        std::runtime_error("RSR Matmul not implemented yet!");
    }
}

void ggml_bitnet_rsr_mul_mat(struct ggml_tensor *dst, const struct ggml_tensor *src0, const struct ggml_tensor *src1) {
    print_once("\n== Using RSR Kernel ==\n");

    // src0: (2560, 2560, 1, 1)
    // src1: (2560, 2, 1, 1)
    // dst: (2560, 2, 1, 1)
    const size_t ne0 = dst->ne[0];
    const size_t ne1 = dst->ne[1];

    float *dst_data = (float *)dst->data;
    for (size_t i = 0; i < ne0 * ne1; i++) {
        dst_data[i] = 0.0f;
    }

    __builtin_debugtrap();
    float *src0_data = (float *)src0->data;
    float *src1_data = (float *)src1->data;

    for (size_t i = 0; i < ne0 * ne1; i++) {
        auto a = src0_data[i];
        cout << a << " -> src0 data" << endl;

        auto b = src1_data[i];
        cout << b << " -> src1 data" << endl;
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
