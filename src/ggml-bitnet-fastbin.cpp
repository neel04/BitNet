#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "ggml.h"
#include "utils.h"

#define QK_I2_S 128
#define QK_I2 128

using namespace std;

template <typename T> using matrix = vector<vector<T>>;

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
static void bitnet_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc) {
    (void)nrc; // Mark parameter as intentionally unused
    (void)by;  // Mark parameter as intentionally unused
    (void)bs;  // Mark parameter as intentionally unused
    (void)bx;  // Mark parameter as intentionally unused

    const uint8_t *x = (uint8_t *)vx;
    const int8_t *y = (int8_t *)vy;

    const int nb = n / QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;

#if defined(__AVX2__)

    __m256i mask = _mm256_set1_epi8(0x03);
    __m256i accu = _mm256_setzero_si256();

    for (int i = 0; i < group32_num; i++) {
        __m256i accu32 = _mm256_setzero_si256();
        for (int j = 0; j < 32; j++) {
            // 128 index
            __m256i xq8_3 = _mm256_loadu_si256((const __m256i *)(x + i * 32 * 32 + j * 32));
            __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
            __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
            __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

            // each 32 index
            xq8_3 = _mm256_and_si256(xq8_3, mask);
            xq8_2 = _mm256_and_si256(xq8_2, mask);
            xq8_1 = _mm256_and_si256(xq8_1, mask);
            xq8_0 = _mm256_and_si256(xq8_0, mask);

            // each 32 index
            __m256i yq8_0 = _mm256_loadu_si256((const __m256i *)(y + i * 128 * 32 + j * 128 + 0));
            __m256i yq8_1 = _mm256_loadu_si256((const __m256i *)(y + i * 128 * 32 + j * 128 + 32));
            __m256i yq8_2 = _mm256_loadu_si256((const __m256i *)(y + i * 128 * 32 + j * 128 + 64));
            __m256i yq8_3 = _mm256_loadu_si256((const __m256i *)(y + i * 128 * 32 + j * 128 + 96));

            // 128 index accumulation add
            // split into 32 accumulation block
            // each block each 128 index accumulated 4index
            // each index maximum 256
            // each block maximum 4 * 256
            // each block accumulation maximum 127 * 256
            // each 32 group index (128 index in one group) needs cast to int32
            xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
            xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
            xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
            xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

            accu32 = _mm256_add_epi16(accu32, _mm256_add_epi16(xq8_0, xq8_1));
            accu32 = _mm256_add_epi16(accu32, _mm256_add_epi16(xq8_2, xq8_3));
        }
        accu = _mm256_add_epi32(_mm256_madd_epi16(accu32, _mm256_set1_epi16(1)), accu);
    }

    for (int i = 0; i < groupla_num; i++) {
        __m256i accula = _mm256_setzero_si256();
        for (int j = 0; j < la_num; j++) {
            // 128 index
            __m256i xq8_3 = _mm256_loadu_si256((const __m256i *)(x + group32_num * 32 * 32 + j * 32));
            __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
            __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
            __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

            // each 32 index
            xq8_3 = _mm256_and_si256(xq8_3, mask);
            xq8_2 = _mm256_and_si256(xq8_2, mask);
            xq8_1 = _mm256_and_si256(xq8_1, mask);
            xq8_0 = _mm256_and_si256(xq8_0, mask);

            // each 32 index
            __m256i yq8_0 = _mm256_loadu_si256((const __m256i *)(y + group32_num * 128 * 32 + j * 128 + 0));
            __m256i yq8_1 = _mm256_loadu_si256((const __m256i *)(y + group32_num * 128 * 32 + j * 128 + 32));
            __m256i yq8_2 = _mm256_loadu_si256((const __m256i *)(y + group32_num * 128 * 32 + j * 128 + 64));
            __m256i yq8_3 = _mm256_loadu_si256((const __m256i *)(y + group32_num * 128 * 32 + j * 128 + 96));

            // 128 index accumulation add
            // split into 32 accumulation block
            // each block each 128 index accumulated 4index
            // each index maximum 256
            // each block maximum 4 * 256
            // each block accumulation maximum 127 * 256
            // each 32 group index (128 index in one group) needs cast to int32
            xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
            xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
            xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
            xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

            accula = _mm256_add_epi16(accula, _mm256_add_epi16(xq8_0, xq8_1));
            accula = _mm256_add_epi16(accula, _mm256_add_epi16(xq8_2, xq8_3));
        }
        accu = _mm256_add_epi32(accu, _mm256_madd_epi16(accula, _mm256_set1_epi16(1)));
    }
    int sumi = hsum_i32_8(accu);
    *s = (float)sumi;

#elif defined(__ARM_NEON)
    // Initialize four 128-bit registers to accumulate results in parallel.
    int32x4_t accu_0 = vdupq_n_s32(0);
    int32x4_t accu_1 = vdupq_n_s32(0);
    int32x4_t accu_2 = vdupq_n_s32(0);
    int32x4_t accu_3 = vdupq_n_s32(0);
    const uint8x16_t mask = vdupq_n_u8(3); // Mask for isolating 2-bit values (0b00000011).

    // Process major blocks of the matrix.
    for (int i = 0; i < group32_num; i++) {

#if defined(__ARM_FEATURE_DOTPROD)

#else
        // Fallback: use 16-bit accumulators for intermediate products.
        int16x8_t accu32_0 = vdupq_n_s16(0);
        int16x8_t accu32_1 = vdupq_n_s16(0);
        int16x8_t accu32_2 = vdupq_n_s16(0);
        int16x8_t accu32_3 = vdupq_n_s16(0);
#endif

        // Process sub-blocks within a major block.
        for (int j = 0; j < 32; j++) {
            // Load 32 bytes, corresponding to 128 2-bit weights.
            uint8x16_t xq8_6 = vld1q_u8(x + i * 32 * 32 + j * 32);
            uint8x16_t xq8_7 = vld1q_u8(x + i * 32 * 32 + j * 32 + 16);

            // Unpack 2-bit values from the loaded bytes by right-shifting.
            uint8x16_t xq8_4 = vshrq_n_u8(xq8_6, 2);
            uint8x16_t xq8_5 = vshrq_n_u8(xq8_7, 2);
            uint8x16_t xq8_2 = vshrq_n_u8(xq8_6, 4);
            uint8x16_t xq8_3 = vshrq_n_u8(xq8_7, 4);
            uint8x16_t xq8_0 = vshrq_n_u8(xq8_6, 6);
            uint8x16_t xq8_1 = vshrq_n_u8(xq8_7, 6);

            // Isolate the lower 2 bits and reinterpret as signed int8 for dot product.
            int8x16_t q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));
            int8x16_t q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
            int8x16_t q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
            int8x16_t q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));
            int8x16_t q8_4 = vreinterpretq_s8_u8(vandq_u8(xq8_4, mask));
            int8x16_t q8_5 = vreinterpretq_s8_u8(vandq_u8(xq8_5, mask));
            int8x16_t q8_6 = vreinterpretq_s8_u8(vandq_u8(xq8_6, mask));
            int8x16_t q8_7 = vreinterpretq_s8_u8(vandq_u8(xq8_7, mask));

            // Load 128 8-bit activation values.
            const int8x16_t yq8_0 = vld1q_s8(y + i * 128 * 32 + j * 128 + 0);
            const int8x16_t yq8_1 = vld1q_s8(y + i * 128 * 32 + j * 128 + 16);
            const int8x16_t yq8_2 = vld1q_s8(y + i * 128 * 32 + j * 128 + 32);
            const int8x16_t yq8_3 = vld1q_s8(y + i * 128 * 32 + j * 128 + 48);
            const int8x16_t yq8_4 = vld1q_s8(y + i * 128 * 32 + j * 128 + 64);
            const int8x16_t yq8_5 = vld1q_s8(y + i * 128 * 32 + j * 128 + 80);
            const int8x16_t yq8_6 = vld1q_s8(y + i * 128 * 32 + j * 128 + 96);
            const int8x16_t yq8_7 = vld1q_s8(y + i * 128 * 32 + j * 128 + 112);

#if defined(__ARM_FEATURE_DOTPROD)
            // Perform dot product and accumulate using dedicated hardware instructions.
            accu_0 = vdotq_s32(accu_0, q8_0, yq8_0);
            accu_1 = vdotq_s32(accu_1, q8_1, yq8_1);
            accu_2 = vdotq_s32(accu_2, q8_2, yq8_2);
            accu_3 = vdotq_s32(accu_3, q8_3, yq8_3);
            accu_0 = vdotq_s32(accu_0, q8_4, yq8_4);
            accu_1 = vdotq_s32(accu_1, q8_5, yq8_5);
            accu_2 = vdotq_s32(accu_2, q8_6, yq8_6);
            accu_3 = vdotq_s32(accu_3, q8_7, yq8_7);
#else
            // Fallback for older ARMv8: multiply and accumulate long, widening to 16-bit.
            accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_0), vget_low_s8(yq8_0));
            accu32_1 = vmlal_s8(accu32_1, vget_high_s8(q8_0), vget_high_s8(yq8_0));
            accu32_2 = vmlal_s8(accu32_2, vget_low_s8(q8_1), vget_low_s8(yq8_1));
            accu32_3 = vmlal_s8(accu32_3, vget_high_s8(q8_1), vget_high_s8(yq8_1));
            accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_2), vget_low_s8(yq8_2));
            accu32_1 = vmlal_s8(accu32_1, vget_high_s8(q8_2), vget_high_s8(yq8_2));
            accu32_2 = vmlal_s8(accu32_2, vget_low_s8(q8_3), vget_low_s8(yq8_3));
            accu32_3 = vmlal_s8(accu32_3, vget_high_s8(q8_3), vget_high_s8(yq8_3));
            accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_4), vget_low_s8(yq8_4));
            accu32_1 = vmlal_s8(accu32_1, vget_high_s8(q8_4), vget_high_s8(yq8_4));
            accu32_2 = vmlal_s8(accu32_2, vget_low_s8(q8_5), vget_low_s8(yq8_5));
            accu32_3 = vmlal_s8(accu32_3, vget_high_s8(q8_5), vget_high_s8(yq8_5));
            accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_6), vget_low_s8(yq8_6));
            accu32_1 = vmlal_s8(accu32_1, vget_high_s8(q8_6), vget_high_s8(yq8_6));
            accu32_2 = vmlal_s8(accu32_2, vget_low_s8(q8_7), vget_low_s8(yq8_7));
            accu32_3 = vmlal_s8(accu32_3, vget_high_s8(q8_7), vget_high_s8(yq8_7));
#endif
        }

#if defined(__ARM_FEATURE_DOTPROD)

#else
        // Accumulate 16-bit intermediate values into 32-bit main accumulators.
        accu_0 = vaddq_s32(accu_0, vmovl_s16(vget_low_s16(accu32_0)));
        accu_0 = vaddq_s32(accu_0, vmovl_high_s16(accu32_0));
        accu_1 = vaddq_s32(accu_1, vmovl_s16(vget_low_s16(accu32_1)));
        accu_1 = vaddq_s32(accu_1, vmovl_high_s16(accu32_1));
        accu_2 = vaddq_s32(accu_2, vmovl_s16(vget_low_s16(accu32_2)));
        accu_2 = vaddq_s32(accu_2, vmovl_high_s16(accu32_2));
        accu_3 = vaddq_s32(accu_3, vmovl_s16(vget_low_s16(accu32_3)));
        accu_3 = vaddq_s32(accu_3, vmovl_high_s16(accu32_3));
#endif
    }

    // Process the remaining blocks that don't fit into a full group of 32.
    for (int i = 0; i < groupla_num; i++) {
#if defined(__ARM_FEATURE_DOTPROD)
#else
        int16x8_t accula_0 = vdupq_n_s16(0);
        int16x8_t accula_1 = vdupq_n_s16(0);
        int16x8_t accula_2 = vdupq_n_s16(0);
        int16x8_t accula_3 = vdupq_n_s16(0);
#endif
        for (int j = 0; j < la_num; j++) {
            uint8x16_t xq8_6 = vld1q_u8(x + group32_num * 32 * 32 + j * 32);
            uint8x16_t xq8_7 = vld1q_u8(x + group32_num * 32 * 32 + j * 32 + 16);
            uint8x16_t xq8_4 = vshrq_n_u8(xq8_6, 2);
            uint8x16_t xq8_5 = vshrq_n_u8(xq8_7, 2);
            uint8x16_t xq8_2 = vshrq_n_u8(xq8_6, 4);
            uint8x16_t xq8_3 = vshrq_n_u8(xq8_7, 4);
            uint8x16_t xq8_0 = vshrq_n_u8(xq8_6, 6);
            uint8x16_t xq8_1 = vshrq_n_u8(xq8_7, 6);

            int8x16_t q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));
            int8x16_t q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
            int8x16_t q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
            int8x16_t q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));
            int8x16_t q8_4 = vreinterpretq_s8_u8(vandq_u8(xq8_4, mask));
            int8x16_t q8_5 = vreinterpretq_s8_u8(vandq_u8(xq8_5, mask));
            int8x16_t q8_6 = vreinterpretq_s8_u8(vandq_u8(xq8_6, mask));
            int8x16_t q8_7 = vreinterpretq_s8_u8(vandq_u8(xq8_7, mask));

            const int8x16_t yq8_0 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 0);
            const int8x16_t yq8_1 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 16);
            const int8x16_t yq8_2 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 32);
            const int8x16_t yq8_3 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 48);
            const int8x16_t yq8_4 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 64);
            const int8x16_t yq8_5 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 80);
            const int8x16_t yq8_6 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 96);
            const int8x16_t yq8_7 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 112);

#if defined(__ARM_FEATURE_DOTPROD)
            accu_0 = vdotq_s32(accu_0, q8_0, yq8_0);
            accu_1 = vdotq_s32(accu_1, q8_1, yq8_1);
            accu_2 = vdotq_s32(accu_2, q8_2, yq8_2);
            accu_3 = vdotq_s32(accu_3, q8_3, yq8_3);
            accu_0 = vdotq_s32(accu_0, q8_4, yq8_4);
            accu_1 = vdotq_s32(accu_1, q8_5, yq8_5);
            accu_2 = vdotq_s32(accu_2, q8_6, yq8_6);
            accu_3 = vdotq_s32(accu_3, q8_7, yq8_7);
#else
            accula_0 = vmlal_s8(accula_0, vget_low_s8(q8_0), vget_low_s8(yq8_0));
            accula_1 = vmlal_s8(accula_1, vget_high_s8(q8_0), vget_high_s8(yq8_0));
            accula_2 = vmlal_s8(accula_2, vget_low_s8(q8_1), vget_low_s8(yq8_1));
            accula_3 = vmlal_s8(accula_3, vget_high_s8(q8_1), vget_high_s8(yq8_1));
            accula_0 = vmlal_s8(accula_0, vget_low_s8(q8_2), vget_low_s8(yq8_2));
            accula_1 = vmlal_s8(accula_1, vget_high_s8(q8_2), vget_high_s8(yq8_2));
            accula_2 = vmlal_s8(accula_2, vget_low_s8(q8_3), vget_low_s8(yq8_3));
            accula_3 = vmlal_s8(accula_3, vget_high_s8(q8_3), vget_high_s8(yq8_3));
            accula_0 = vmlal_s8(accula_0, vget_low_s8(q8_4), vget_low_s8(yq8_4));
            accula_1 = vmlal_s8(accula_1, vget_high_s8(q8_4), vget_high_s8(yq8_4));
            accula_2 = vmlal_s8(accula_2, vget_low_s8(q8_5), vget_low_s8(yq8_5));
            accula_3 = vmlal_s8(accula_3, vget_high_s8(q8_5), vget_high_s8(yq8_5));
            accula_0 = vmlal_s8(accula_0, vget_low_s8(q8_6), vget_low_s8(yq8_6));
            accula_1 = vmlal_s8(accula_1, vget_high_s8(q8_6), vget_high_s8(yq8_6));
            accula_2 = vmlal_s8(accula_2, vget_low_s8(q8_7), vget_low_s8(yq8_7));
            accula_3 = vmlal_s8(accula_3, vget_high_s8(q8_7), vget_high_s8(yq8_7));
#endif
        }
#if defined(__ARM_FEATURE_DOTPROD)

#else
        accu_0 = vaddq_s32(accu_0, vmovl_s16(vget_low_s16(accula_0)));
        accu_0 = vaddq_s32(accu_0, vmovl_high_s16(accula_0));
        accu_1 = vaddq_s32(accu_1, vmovl_s16(vget_low_s16(accula_1)));
        accu_1 = vaddq_s32(accu_1, vmovl_high_s16(accula_1));
        accu_2 = vaddq_s32(accu_2, vmovl_s16(vget_low_s16(accula_2)));
        accu_2 = vaddq_s32(accu_2, vmovl_high_s16(accula_2));
        accu_3 = vaddq_s32(accu_3, vmovl_s16(vget_low_s16(accula_3)));
        accu_3 = vaddq_s32(accu_3, vmovl_high_s16(accula_3));
#endif
    }
    // Horizontally add the final accumulator vectors to get the single dot product result.
    accu_0 = vaddq_s32(accu_0, accu_1);
    accu_2 = vaddq_s32(accu_2, accu_3); // compliler parallelizes
    accu_0 = vaddq_s32(accu_0, accu_2);
    int sumi = vaddlvq_s32(accu_0);
    *s = (float)sumi;

#endif
}

static void preprocess_weights(const uint8x16_t &mask,
                               const int &group32_num,
                               const int &la_num,
                               const int &groupla_num,
                               const uint8_t *&quant_weight_ptr,
                               uint8_t *array_buffer,
                               uint8_t *binary_buffer_0,
                               uint8_t *binary_buffer_1) {

    int8x16_t ones = vdupq_n_u8(1);
    int8x16_t negs = vdupq_n_u8(-1);

    for (int i = 0; i < group32_num; i++) {
        for (int j = 0; j < 32; j++) {
            // 32 bytes <==> 128x 2-bit weights
            uint8x16_t xq8_6 = vld1q_u8(quant_weight_ptr + i * 32 * 32 + j * 32);      // first 16
            uint8x16_t xq8_7 = vld1q_u8(quant_weight_ptr + i * 32 * 32 + j * 32 + 16); // rest

            uint8x16_t xq8_4 = vshrq_n_u8(xq8_6, 2);
            uint8x16_t xq8_5 = vshrq_n_u8(xq8_7, 2);
            uint8x16_t xq8_2 = vshrq_n_u8(xq8_6, 4);
            uint8x16_t xq8_3 = vshrq_n_u8(xq8_7, 4);
            uint8x16_t xq8_0 = vshrq_n_u8(xq8_6, 6);
            uint8x16_t xq8_1 = vshrq_n_u8(xq8_7, 6);

            // Isolate the lower 2 bits and reinterpret as signed int8 for dot product
            int8x16_t q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));
            int8x16_t q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
            int8x16_t q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
            int8x16_t q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));
            int8x16_t q8_4 = vreinterpretq_s8_u8(vandq_u8(xq8_4, mask));
            int8x16_t q8_5 = vreinterpretq_s8_u8(vandq_u8(xq8_5, mask));
            int8x16_t q8_6 = vreinterpretq_s8_u8(vandq_u8(xq8_6, mask));
            int8x16_t q8_7 = vreinterpretq_s8_u8(vandq_u8(xq8_7, mask));

            // Also mask it to 0/1 at the end
            int8x16_t b8_0_0 = vandq_u8(vceqq_s8(q8_0 - 1, ones), vdupq_n_u8(1));
            int8x16_t b8_1_0 = vandq_u8(vceqq_s8(q8_1 - 1, ones), vdupq_n_u8(1));
            int8x16_t b8_2_0 = vandq_u8(vceqq_s8(q8_2 - 1, ones), vdupq_n_u8(1));
            int8x16_t b8_3_0 = vandq_u8(vceqq_s8(q8_3 - 1, ones), vdupq_n_u8(1));
            int8x16_t b8_4_0 = vandq_u8(vceqq_s8(q8_4 - 1, ones), vdupq_n_u8(1));
            int8x16_t b8_5_0 = vandq_u8(vceqq_s8(q8_5 - 1, ones), vdupq_n_u8(1));
            int8x16_t b8_6_0 = vandq_u8(vceqq_s8(q8_6 - 1, ones), vdupq_n_u8(1));
            int8x16_t b8_7_0 = vandq_u8(vceqq_s8(q8_7 - 1, ones), vdupq_n_u8(1));

            int8x16_t b8_0_1 = vandq_u8(vceqq_s8(q8_0 - 1, negs), vdupq_n_u8(1));
            int8x16_t b8_1_1 = vandq_u8(vceqq_s8(q8_1 - 1, negs), vdupq_n_u8(1));
            int8x16_t b8_2_1 = vandq_u8(vceqq_s8(q8_2 - 1, negs), vdupq_n_u8(1));
            int8x16_t b8_3_1 = vandq_u8(vceqq_s8(q8_3 - 1, negs), vdupq_n_u8(1));
            int8x16_t b8_4_1 = vandq_u8(vceqq_s8(q8_4 - 1, negs), vdupq_n_u8(1));
            int8x16_t b8_5_1 = vandq_u8(vceqq_s8(q8_5 - 1, negs), vdupq_n_u8(1));
            int8x16_t b8_6_1 = vandq_u8(vceqq_s8(q8_6 - 1, negs), vdupq_n_u8(1));
            int8x16_t b8_7_1 = vandq_u8(vceqq_s8(q8_7 - 1, negs), vdupq_n_u8(1));

            // Repack the unpacked 2-bit values back into bytes
            uint8x16_t pack_a =
                vorrq_u8(vorrq_u8(vorrq_u8(vshlq_n_u8(xq8_0, 6), vshlq_n_u8(xq8_2, 4)), vshlq_n_u8(xq8_4, 2)), xq8_6);

            uint8x16_t pack_b =
                vorrq_u8(vorrq_u8(vorrq_u8(vshlq_n_u8(xq8_1, 6), vshlq_n_u8(xq8_3, 4)), vshlq_n_u8(xq8_5, 2)), xq8_7);

            uint8x16_t pack_bin_0_a = vorrq_u8(
                vorrq_u8(vorrq_u8(vshlq_n_u8(b8_0_0, 6), vshlq_n_u8(b8_2_0, 4)), vshlq_n_u8(b8_4_0, 2)), b8_6_0);

            uint8x16_t pack_bin_0_b = vorrq_u8(
                vorrq_u8(vorrq_u8(vshlq_n_u8(b8_1_0, 6), vshlq_n_u8(b8_3_0, 4)), vshlq_n_u8(b8_5_0, 2)), b8_7_0);

            uint8x16_t pack_bin_1_a = vorrq_u8(
                vorrq_u8(vorrq_u8(vshlq_n_u8(b8_0_1, 6), vshlq_n_u8(b8_2_1, 4)), vshlq_n_u8(b8_4_1, 2)), b8_6_1);

            uint8x16_t pack_bin_1_b = vorrq_u8(
                vorrq_u8(vorrq_u8(vshlq_n_u8(b8_1_1, 6), vshlq_n_u8(b8_3_1, 4)), vshlq_n_u8(b8_5_1, 2)), b8_7_1);

            // Store to array_buffer with same offset structure
            uint8_t *buffer_ptr = (uint8_t *)array_buffer + (i * 32 * 32 + j * 32);
            uint8_t *bin_0_ptr = (uint8_t *)binary_buffer_0 + (i * 32 * 32 + j * 32);
            uint8_t *bin_1_ptr = (uint8_t *)binary_buffer_1 + (i * 32 * 32 + j * 32);

            vst1q_u8(bin_0_ptr, pack_bin_0_a);
            vst1q_u8(bin_0_ptr + 16, pack_bin_0_b);

            vst1q_u8(bin_1_ptr, pack_bin_1_a);
            vst1q_u8(bin_1_ptr + 16, pack_bin_1_b);

            vst1q_u8(buffer_ptr, pack_a);
            vst1q_u8(buffer_ptr + 16, pack_b);
        }
    }

    for (int i = 0; i < groupla_num; i++) {
        for (int j = 0; j < la_num; j++) {
            uint8x16_t xq8_6 = vld1q_u8(quant_weight_ptr + group32_num * 32 * 32 + j * 32);
            uint8x16_t xq8_7 = vld1q_u8(quant_weight_ptr + group32_num * 32 * 32 + j * 32 + 16);

            uint8x16_t xq8_4 = vshrq_n_u8(xq8_6, 2);
            uint8x16_t xq8_5 = vshrq_n_u8(xq8_7, 2);
            uint8x16_t xq8_2 = vshrq_n_u8(xq8_6, 4);
            uint8x16_t xq8_3 = vshrq_n_u8(xq8_7, 4);
            uint8x16_t xq8_0 = vshrq_n_u8(xq8_6, 6);
            uint8x16_t xq8_1 = vshrq_n_u8(xq8_7, 6);

            // Isolate the lower 2 bits and reinterpret as signed int8 for dot product
            int8x16_t q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));
            int8x16_t q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
            int8x16_t q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
            int8x16_t q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));
            int8x16_t q8_4 = vreinterpretq_s8_u8(vandq_u8(xq8_4, mask));
            int8x16_t q8_5 = vreinterpretq_s8_u8(vandq_u8(xq8_5, mask));
            int8x16_t q8_6 = vreinterpretq_s8_u8(vandq_u8(xq8_6, mask));
            int8x16_t q8_7 = vreinterpretq_s8_u8(vandq_u8(xq8_7, mask));

            // Also mask it to 0/1 at the end
            int8x16_t b8_0_0 = vandq_u8(vceqq_s8(q8_0 - 1, ones), vdupq_n_u8(1));
            int8x16_t b8_1_0 = vandq_u8(vceqq_s8(q8_1 - 1, ones), vdupq_n_u8(1));
            int8x16_t b8_2_0 = vandq_u8(vceqq_s8(q8_2 - 1, ones), vdupq_n_u8(1));
            int8x16_t b8_3_0 = vandq_u8(vceqq_s8(q8_3 - 1, ones), vdupq_n_u8(1));
            int8x16_t b8_4_0 = vandq_u8(vceqq_s8(q8_4 - 1, ones), vdupq_n_u8(1));
            int8x16_t b8_5_0 = vandq_u8(vceqq_s8(q8_5 - 1, ones), vdupq_n_u8(1));
            int8x16_t b8_6_0 = vandq_u8(vceqq_s8(q8_6 - 1, ones), vdupq_n_u8(1));
            int8x16_t b8_7_0 = vandq_u8(vceqq_s8(q8_7 - 1, ones), vdupq_n_u8(1));

            int8x16_t b8_0_1 = vandq_u8(vceqq_s8(q8_0 - 1, negs), vdupq_n_u8(1));
            int8x16_t b8_1_1 = vandq_u8(vceqq_s8(q8_1 - 1, negs), vdupq_n_u8(1));
            int8x16_t b8_2_1 = vandq_u8(vceqq_s8(q8_2 - 1, negs), vdupq_n_u8(1));
            int8x16_t b8_3_1 = vandq_u8(vceqq_s8(q8_3 - 1, negs), vdupq_n_u8(1));
            int8x16_t b8_4_1 = vandq_u8(vceqq_s8(q8_4 - 1, negs), vdupq_n_u8(1));
            int8x16_t b8_5_1 = vandq_u8(vceqq_s8(q8_5 - 1, negs), vdupq_n_u8(1));
            int8x16_t b8_6_1 = vandq_u8(vceqq_s8(q8_6 - 1, negs), vdupq_n_u8(1));
            int8x16_t b8_7_1 = vandq_u8(vceqq_s8(q8_7 - 1, negs), vdupq_n_u8(1));


            // Repack the unpacked 2-bit values back into bytes
            uint8x16_t pack_a =
                vorrq_u8(vorrq_u8(vorrq_u8(vshlq_n_u8(xq8_0, 6), vshlq_n_u8(xq8_2, 4)), vshlq_n_u8(xq8_4, 2)), xq8_6);

            uint8x16_t pack_b =
                vorrq_u8(vorrq_u8(vorrq_u8(vshlq_n_u8(xq8_1, 6), vshlq_n_u8(xq8_3, 4)), vshlq_n_u8(xq8_5, 2)), xq8_7);

            uint8x16_t pack_bin_0_a = vorrq_u8(
                vorrq_u8(vorrq_u8(vshlq_n_u8(b8_0_0, 6), vshlq_n_u8(b8_2_0, 4)), vshlq_n_u8(b8_4_0, 2)), b8_6_0);

            uint8x16_t pack_bin_0_b = vorrq_u8(
                vorrq_u8(vorrq_u8(vshlq_n_u8(b8_1_0, 6), vshlq_n_u8(b8_3_0, 4)), vshlq_n_u8(b8_5_0, 2)), b8_7_0);

            uint8x16_t pack_bin_1_a = vorrq_u8(
                vorrq_u8(vorrq_u8(vshlq_n_u8(b8_0_1, 6), vshlq_n_u8(b8_2_1, 4)), vshlq_n_u8(b8_4_1, 2)), b8_6_1);

            uint8x16_t pack_bin_1_b = vorrq_u8(
                vorrq_u8(vorrq_u8(vshlq_n_u8(b8_1_1, 6), vshlq_n_u8(b8_3_1, 4)), vshlq_n_u8(b8_5_1, 2)), b8_7_1);

            // Store to array_buffer with same offset structure
            uint8_t *buffer_ptr = (uint8_t *)array_buffer + (group32_num * 32 * 32 + j * 32);
            uint8_t *bin_0_ptr = (uint8_t *)binary_buffer_0 + (group32_num * 32 * 32 + j * 32);
            uint8_t *bin_1_ptr = (uint8_t *)binary_buffer_1 + (group32_num * 32 * 32 + j * 32);

            vst1q_u8(bin_0_ptr, pack_bin_0_a);
            vst1q_u8(bin_0_ptr + 16, pack_bin_0_b);

            vst1q_u8(bin_1_ptr, pack_bin_1_a);
            vst1q_u8(bin_1_ptr + 16, pack_bin_1_b);

            vst1q_u8(buffer_ptr, pack_a);
            vst1q_u8(buffer_ptr + 16, pack_b);
        }
    }
}

extern "C" {
void ggml_bitnet_fastbin_mul_mat(const struct ggml_tensor *src0,
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

    print_once("\n== Using Fast-Binary Kernel ====\n");

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
    blck_0 = blck_1 = CHUNK_SIZE; // FIX: Hardcoded chunk size

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
                // Should by always equal to `chunk_size` however
                const int output_rows = std::min(blck_0, (int)(ir0_end - iir0));
                assert(output_rows <= (int)CHUNK_SIZE);

                float tmp[CHUNK_SIZE];
                float bin1[CHUNK_SIZE];
                float bin2[CHUNK_SIZE];

                const uint8x16_t mask = vdupq_n_u8(3); // last 2 bits

                // Timer t("Fast Bin");

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    const int nb = ne00 / QK_I2_S;
                    const int group32_num = nb / 32;
                    const int la_num = nb % 32;
                    const int groupla_num = nb % 32 != 0 ? 1 : 0;

                    auto quant_weight_ptr = src0_row + ir0 * nb01 / 4;

                    uint8_t array_buffer[6912];

                    uint8_t binary_buffer_0[6912];
                    uint8_t binary_buffer_1[6912];

                    preprocess_weights(mask,
                                       group32_num,
                                       la_num,
                                       groupla_num,
                                       quant_weight_ptr,
                                       array_buffer,
                                       binary_buffer_0,
                                       binary_buffer_1);

                    if (src0->type == GGML_TYPE_I2_S) {
                        bitnet_vec_dot(ne00,
                                       &bin1[ir0 - iir0],
                                       (num_rows_per_vec_dot > 1 ? 16 : 0),
                                       (uint8_t *)binary_buffer_0,
                                       (num_rows_per_vec_dot > 1 ? nb01 : 0),
                                       src1_col_de,
                                       (num_rows_per_vec_dot > 1 ? src1_col_stride : 0),
                                       num_rows_per_vec_dot);

                        bitnet_vec_dot(ne00,
                                       &bin2[ir0 - iir0],
                                       (num_rows_per_vec_dot > 1 ? 16 : 0),
                                       (uint8_t *)binary_buffer_1,
                                       (num_rows_per_vec_dot > 1 ? nb01 : 0),
                                       src1_col_de,
                                       (num_rows_per_vec_dot > 1 ? src1_col_stride : 0),
                                       num_rows_per_vec_dot);

                        tmp[ir0 - iir0] = (bin1[ir0 - iir0] - bin2[ir0 - iir0]) / (act_scales[i1]) * (*scale);
                    } else {
                        // Fallback to default GGML Kernel to handle higher precision gemv
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
                // t.stop();

                memcpy(&dst_col[iir0], tmp, output_rows * sizeof(float));
            }
        }
    }
}
}
