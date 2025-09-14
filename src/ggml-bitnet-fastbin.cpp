#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
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

#if defined(__ARM_NEON)
static inline int32x4_t reduce_i8x16_to_i32x4_simd(int8x16_t v) {
    int16x8_t p16 = vpaddlq_s8(v);
    int32x4_t sums4 = vpaddlq_s16(p16);
    return sums4;
}

static inline int32x4_t reduce_i8x8_to_i32x4_simd(int8x8_t v) {
    int16x8_t p16 = vmovl_s8(v);        // Widen int8x8_t to int16x8_t
    int32x4_t sums4 = vpaddlq_s16(p16); // Pairwise add to int32x4_t
    return sums4;
}
#endif

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

    const int nb = n / 64;
    const int group16_num = nb / 16;
    const int la_num = nb % 16;
    const int groupla_num = nb % 16 != 0 ? 1 : 0;

#if defined(__ARM_NEON)
    // Initialize four 128-bit registers to accumulate results in parallel.
    int32x4_t accu_0 = vdupq_n_s32(0);
    int32x4_t accu_1 = vdupq_n_s32(0);
    int32x4_t accu_2 = vdupq_n_s32(0);
    int32x4_t accu_3 = vdupq_n_s32(0);

    const uint8x8_t mask = vdup_n_u8(1); // Mask for isolating 1-bit values (0b00000001).

    // Process major blocks of the matrix.
    for (int i = 0; i < group16_num; i++) {
#if defined(__ARM_FEATURE_DOTPROD)

#else
        // Fallback: use 16-bit accumulators for intermediate products.
        int16x8_t accu32_0 = vdupq_n_s16(0);
        int16x8_t accu32_1 = vdupq_n_s16(0);
        int16x8_t accu32_2 = vdupq_n_s16(0);
        int16x8_t accu32_3 = vdupq_n_s16(0);
#endif

        // Process sub-blocks within a major block.
        for (int j = 0; j < 16; j++) {
            // Load 8 bytes for 64 densely packed binary weights
            uint8x8_t xq8 = vld1_u8(x + i * 16 * 8 + j * 8);

            // DENSE: 6 4 2 0 - 7 5 3 1
            // Unpack 1-bit values from the loaded bytes by right-shifting.
            uint8x8_t xq8_1 = xq8;
            uint8x8_t xq8_3 = xq8 >> 1;
            uint8x8_t xq8_5 = xq8 >> 2;
            uint8x8_t xq8_7 = xq8 >> 3;
            uint8x8_t xq8_0 = xq8 >> 4;
            uint8x8_t xq8_2 = xq8 >> 5;
            uint8x8_t xq8_4 = xq8 >> 6;
            uint8x8_t xq8_6 = xq8 >> 7;

            // Isolate the lower bit and reinterpret as signed int8 for dot product.
            int8x8_t q8_0 = vreinterpret_s8_u8(vand_u8(xq8_0, mask));
            int8x8_t q8_1 = vreinterpret_s8_u8(vand_u8(xq8_1, mask));
            int8x8_t q8_2 = vreinterpret_s8_u8(vand_u8(xq8_2, mask));
            int8x8_t q8_3 = vreinterpret_s8_u8(vand_u8(xq8_3, mask));
            int8x8_t q8_4 = vreinterpret_s8_u8(vand_u8(xq8_4, mask));
            int8x8_t q8_5 = vreinterpret_s8_u8(vand_u8(xq8_5, mask));
            int8x8_t q8_6 = vreinterpret_s8_u8(vand_u8(xq8_6, mask));
            int8x8_t q8_7 = vreinterpret_s8_u8(vand_u8(xq8_7, mask));

            // Load 64 8-bit activation values (dense): 4x16 contiguous blocks
            const int8x16_t yq8_0 = vld1q_s8(y + i * 64 * 16 + j * 64 + 0);
            const int8x16_t yq8_1 = vld1q_s8(y + i * 64 * 16 + j * 64 + 16);
            const int8x16_t yq8_2 = vld1q_s8(y + i * 64 * 16 + j * 64 + 32);
            const int8x16_t yq8_3 = vld1q_s8(y + i * 64 * 16 + j * 64 + 48);

#if defined(__ARM_FEATURE_DOTPROD)
            // Use dot-product to sum selected activations (0/1 mask) directly
            const int8x16_t m0 = vcombine_s8(q8_0, q8_1);
            const int8x16_t m1 = vcombine_s8(q8_2, q8_3);
            const int8x16_t m2 = vcombine_s8(q8_4, q8_5);
            const int8x16_t m3 = vcombine_s8(q8_6, q8_7);

            accu_0 = vdotq_s32(accu_0, yq8_0, m0);
            accu_1 = vdotq_s32(accu_1, yq8_1, m1);
            accu_2 = vdotq_s32(accu_2, yq8_2, m2);
            accu_3 = vdotq_s32(accu_3, yq8_3, m3);
#else
            // Fallback for older ARMv8: multiply and accumulate long, widening to 16-bit.
            // Dense format: Weight vectors (q8_*) are int8x8_t, activation vectors (yq8_*) are int8x16_t
            accu32_0 = vmlal_s8(accu32_0, q8_0, vget_low_s8(yq8_0));
            accu32_1 = vmlal_s8(accu32_1, q8_1, vget_high_s8(yq8_0));
            accu32_2 = vmlal_s8(accu32_2, q8_2, vget_low_s8(yq8_1));
            accu32_3 = vmlal_s8(accu32_3, q8_3, vget_high_s8(yq8_1));
            accu32_0 = vmlal_s8(accu32_0, q8_4, vget_low_s8(yq8_2));
            accu32_1 = vmlal_s8(accu32_1, q8_5, vget_high_s8(yq8_2));
            accu32_2 = vmlal_s8(accu32_2, q8_6, vget_low_s8(yq8_3));
            accu32_3 = vmlal_s8(accu32_3, q8_7, vget_high_s8(yq8_3));
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

    // Process the remaining blocks that don't fit into a full group of 16.
    for (int i = 0; i < groupla_num; i++) {
#if defined(__ARM_FEATURE_DOTPROD)
#else
        int16x8_t accula_0 = vdupq_n_s16(0);
        int16x8_t accula_1 = vdupq_n_s16(0);
        int16x8_t accula_2 = vdupq_n_s16(0);
        int16x8_t accula_3 = vdupq_n_s16(0);
#endif
        for (int j = 0; j < la_num; j++) {
            // Load 8 bytes for 64 densely packed binary weights
            // Correct remainder base/stride: 8 bytes per 64 weights
            uint8x8_t xq8 = vld1_u8(x + group16_num * 16 * 8 + j * 8);

            // DENSE: 6 4 2 0 - 7 5 3 1
            // Unpack 1-bit values from the loaded bytes by right-shifting.
            uint8x8_t xq8_1 = xq8;
            uint8x8_t xq8_3 = xq8 >> 1;
            uint8x8_t xq8_5 = xq8 >> 2;
            uint8x8_t xq8_7 = xq8 >> 3;
            uint8x8_t xq8_0 = xq8 >> 4;
            uint8x8_t xq8_2 = xq8 >> 5;
            uint8x8_t xq8_4 = xq8 >> 6;
            uint8x8_t xq8_6 = xq8 >> 7;

            // Isolate the lower bit and reinterpret as signed int8 for dot product.
            int8x8_t q8_0 = vreinterpret_s8_u8(vand_u8(xq8_0, mask));
            int8x8_t q8_1 = vreinterpret_s8_u8(vand_u8(xq8_1, mask));
            int8x8_t q8_2 = vreinterpret_s8_u8(vand_u8(xq8_2, mask));
            int8x8_t q8_3 = vreinterpret_s8_u8(vand_u8(xq8_3, mask));
            int8x8_t q8_4 = vreinterpret_s8_u8(vand_u8(xq8_4, mask));
            int8x8_t q8_5 = vreinterpret_s8_u8(vand_u8(xq8_5, mask));
            int8x8_t q8_6 = vreinterpret_s8_u8(vand_u8(xq8_6, mask));
            int8x8_t q8_7 = vreinterpret_s8_u8(vand_u8(xq8_7, mask));

            // Load 64 8-bit activations. Base on completed full groups, not local loop index
            const int base_y = group16_num * 64 * 16;
            const int8x16_t yq8_0 = vld1q_s8(y + base_y + j * 64 + 0);
            const int8x16_t yq8_1 = vld1q_s8(y + base_y + j * 64 + 16);
            const int8x16_t yq8_2 = vld1q_s8(y + base_y + j * 64 + 32);
            const int8x16_t yq8_3 = vld1q_s8(y + base_y + j * 64 + 48);

#if defined(__ARM_FEATURE_DOTPROD)
            const int8x16_t m0 = vcombine_s8(q8_0, q8_1);
            const int8x16_t m1 = vcombine_s8(q8_2, q8_3);
            const int8x16_t m2 = vcombine_s8(q8_4, q8_5);
            const int8x16_t m3 = vcombine_s8(q8_6, q8_7);

            accu_0 = vdotq_s32(accu_0, yq8_0, m0);
            accu_1 = vdotq_s32(accu_1, yq8_1, m1);
            accu_2 = vdotq_s32(accu_2, yq8_2, m2);
            accu_3 = vdotq_s32(accu_3, yq8_3, m3);
#else
            // Fallback for older ARMv8: multiply and accumulate long, widening to 16-bit.
            // Dense format: Weight vectors (q8_*) are int8x8_t, activation vectors (yq8_*) are int8x16_t
            accula_0 = vmlal_s8(accula_0, q8_0, vget_low_s8(yq8_0));
            accula_1 = vmlal_s8(accula_1, q8_1, vget_high_s8(yq8_0));
            accula_2 = vmlal_s8(accula_2, q8_2, vget_low_s8(yq8_1));
            accula_3 = vmlal_s8(accula_3, q8_3, vget_high_s8(yq8_1));
            accula_0 = vmlal_s8(accula_0, q8_4, vget_low_s8(yq8_2));
            accula_1 = vmlal_s8(accula_1, q8_5, vget_high_s8(yq8_2));
            accula_2 = vmlal_s8(accula_2, q8_6, vget_low_s8(yq8_3));
            accula_3 = vmlal_s8(accula_3, q8_7, vget_high_s8(yq8_3));
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
    accu_2 = vaddq_s32(accu_2, accu_3);
    accu_0 = vaddq_s32(accu_0, accu_2);
    int accumi = vaddlvq_s32(accu_0);
    *s = (float)accumi;

#elif defined(__AVX2__)
    // AVX2/SSE4.1 implementation using maddubs + madd
    __m128i acc32 = _mm_setzero_si128();
    const __m128i ones16 = _mm_set1_epi16(1);
    const __m128i one16 = _mm_set1_epi16(1);
    const __m128i zero = _mm_setzero_si128();

    for (int i = 0; i < group16_num; ++i) {
        for (int j = 0; j < 16; ++j) {
            // Load 8 bytes (64 weights packed) for this sub-block
            const uint8_t *xb = x + i * 16 * 8 + j * 8;
            __m128i xq8 = _mm_loadl_epi64((const __m128i *)xb);
            __m128i x16 = _mm_unpacklo_epi8(xq8, zero); // 8 lanes of uint16

            // Build 0/1 masks for eight bit positions per lane
            __m128i b1 = _mm_and_si128(_mm_srli_epi16(x16, 0), one16); // xq8_1
            __m128i b3 = _mm_and_si128(_mm_srli_epi16(x16, 1), one16); // xq8_3
            __m128i b5 = _mm_and_si128(_mm_srli_epi16(x16, 2), one16); // xq8_5
            __m128i b7 = _mm_and_si128(_mm_srli_epi16(x16, 3), one16); // xq8_7
            __m128i b0 = _mm_and_si128(_mm_srli_epi16(x16, 4), one16); // xq8_0
            __m128i b2 = _mm_and_si128(_mm_srli_epi16(x16, 5), one16); // xq8_2
            __m128i b4 = _mm_and_si128(_mm_srli_epi16(x16, 6), one16); // xq8_4
            __m128i b6 = _mm_and_si128(_mm_srli_epi16(x16, 7), one16); // xq8_6

            // Concatenate into 16-byte masks matching y blocks
            __m128i m0 = _mm_packs_epi16(b0, b1);
            __m128i m1 = _mm_packs_epi16(b2, b3);
            __m128i m2 = _mm_packs_epi16(b4, b5);
            __m128i m3 = _mm_packs_epi16(b6, b7);

            // Load 64 activations: 4x16 blocks
            const int base_y = i * 64 * 16 + j * 64;
            __m128i y0 = _mm_loadu_si128((const __m128i *)(y + base_y + 0));
            __m128i y1 = _mm_loadu_si128((const __m128i *)(y + base_y + 16));
            __m128i y2 = _mm_loadu_si128((const __m128i *)(y + base_y + 32));
            __m128i y3 = _mm_loadu_si128((const __m128i *)(y + base_y + 48));

            // Sum selected elements: maddubs (u8* s8 -> s16 pairwise sums) then madd (s16*1 -> s32 sums)
            __m128i p0 = _mm_maddubs_epi16(m0, y0);
            __m128i p1 = _mm_maddubs_epi16(m1, y1);
            __m128i p2 = _mm_maddubs_epi16(m2, y2);
            __m128i p3 = _mm_maddubs_epi16(m3, y3);
            __m128i t0 = _mm_madd_epi16(p0, ones16);
            __m128i t1 = _mm_madd_epi16(p1, ones16);
            __m128i t2 = _mm_madd_epi16(p2, ones16);
            __m128i t3 = _mm_madd_epi16(p3, ones16);
            acc32 = _mm_add_epi32(acc32, t0);
            acc32 = _mm_add_epi32(acc32, t1);
            acc32 = _mm_add_epi32(acc32, t2);
            acc32 = _mm_add_epi32(acc32, t3);
        }
    }

    // Remainder blocks
    for (int i = 0; i < groupla_num; ++i) {
        for (int j = 0; j < la_num; ++j) {
            const uint8_t *xb = x + group16_num * 16 * 8 + j * 8;
            __m128i xq8 = _mm_loadl_epi64((const __m128i *)xb);
            __m128i x16 = _mm_unpacklo_epi8(xq8, zero);

            __m128i b1 = _mm_and_si128(_mm_srli_epi16(x16, 0), one16);
            __m128i b3 = _mm_and_si128(_mm_srli_epi16(x16, 1), one16);
            __m128i b5 = _mm_and_si128(_mm_srli_epi16(x16, 2), one16);
            __m128i b7 = _mm_and_si128(_mm_srli_epi16(x16, 3), one16);
            __m128i b0 = _mm_and_si128(_mm_srli_epi16(x16, 4), one16);
            __m128i b2 = _mm_and_si128(_mm_srli_epi16(x16, 5), one16);
            __m128i b4 = _mm_and_si128(_mm_srli_epi16(x16, 6), one16);
            __m128i b6 = _mm_and_si128(_mm_srli_epi16(x16, 7), one16);

            __m128i m0 = _mm_packs_epi16(b0, b1);
            __m128i m1 = _mm_packs_epi16(b2, b3);
            __m128i m2 = _mm_packs_epi16(b4, b5);
            __m128i m3 = _mm_packs_epi16(b6, b7);

            const int base_y = group16_num * 64 * 16 + j * 64;
            __m128i y0 = _mm_loadu_si128((const __m128i *)(y + base_y + 0));
            __m128i y1 = _mm_loadu_si128((const __m128i *)(y + base_y + 16));
            __m128i y2 = _mm_loadu_si128((const __m128i *)(y + base_y + 32));
            __m128i y3 = _mm_loadu_si128((const __m128i *)(y + base_y + 48));

            __m128i p0 = _mm_maddubs_epi16(m0, y0);
            __m128i p1 = _mm_maddubs_epi16(m1, y1);
            __m128i p2 = _mm_maddubs_epi16(m2, y2);
            __m128i p3 = _mm_maddubs_epi16(m3, y3);
            __m128i t0 = _mm_madd_epi16(p0, ones16);
            __m128i t1 = _mm_madd_epi16(p1, ones16);
            __m128i t2 = _mm_madd_epi16(p2, ones16);
            __m128i t3 = _mm_madd_epi16(p3, ones16);
            acc32 = _mm_add_epi32(acc32, t0);
            acc32 = _mm_add_epi32(acc32, t1);
            acc32 = _mm_add_epi32(acc32, t2);
            acc32 = _mm_add_epi32(acc32, t3);
        }
    }

    // Horizontal sum of 4 lanes
    __m128i shuf = _mm_shuffle_epi32(acc32, _MM_SHUFFLE(2, 3, 0, 1));
    acc32 = _mm_add_epi32(acc32, shuf);
    shuf = _mm_shuffle_epi32(acc32, _MM_SHUFFLE(1, 0, 3, 2));
    acc32 = _mm_add_epi32(acc32, shuf);
    int accumi = _mm_cvtsi128_si32(acc32);
    *s = (float)accumi;

#else
    // Portable fallback: iterate bits and sum activations
    int sum = 0;

    for (int c = 0; c < nb; ++c) {
        const uint8_t *bits = x + (size_t)c * 8; // 8 bytes per 64 values
        const int8_t *yy = y + (size_t)c * 64;

        for (int sidx = 0; sidx < 64; ++sidx) {
            int g = sidx / 16;             // 0..3
            int h = (sidx / 8) & 1;        // 0 or 1
            int out_bit = h ? g : (4 + g); // 0..7 per DENSE mapping
            int out_byte = sidx & 7;       // sidx % 8
            uint8_t sel = (uint8_t)((bits[out_byte] >> out_bit) & 1u);
            sum += (int)sel * (int)yy[sidx];
        }
    }

    *s = (float)sum;
#endif
}

static void bitnet_vec_dot_fused(int n,
                                 float *s,
                                 size_t bs,
                                 const void *vx_pos,
                                 size_t bx_pos,
                                 const void *vx_neg,
                                 size_t bx_neg,
                                 const void *vy,
                                 size_t by,
                                 int nrc) {
    (void)nrc; // unused
    (void)by;
    (void)bs;
    (void)bx_pos;
    (void)bx_neg;

    const uint8_t *xpos = (const uint8_t *)vx_pos;
    const uint8_t *xneg = (const uint8_t *)vx_neg;
    const int8_t *y = (const int8_t *)vy;

    const int nb = n / 64;           // 8 bytes per 64 weights
    const int group16_num = nb / 16; // full groups of 16 sub-blocks
    const int la_num = nb % 16;      // leftover sub-blocks
    const int groupla_num = la_num != 0 ? 1 : 0;

#if defined(__ARM_NEON)
    int32x4_t accu_0 = vdupq_n_s32(0);
    int32x4_t accu_1 = vdupq_n_s32(0);
    int32x4_t accu_2 = vdupq_n_s32(0);
    int32x4_t accu_3 = vdupq_n_s32(0);

    const uint8x8_t mask = vdup_n_u8(1);

    for (int i = 0; i < group16_num; ++i) {
#if defined(__ARM_FEATURE_DOTPROD)
        // Prefer DOTPROD path
#else
        int16x8_t acc16_0 = vdupq_n_s16(0);
        int16x8_t acc16_1 = vdupq_n_s16(0);
        int16x8_t acc16_2 = vdupq_n_s16(0);
        int16x8_t acc16_3 = vdupq_n_s16(0);
#endif
        for (int j = 0; j < 16; ++j) {
            // Load 8 packed bytes for pos/neg
            uint8x8_t xq8p = vld1_u8(xpos + i * 16 * 8 + j * 8);
            uint8x8_t xq8n = vld1_u8(xneg + i * 16 * 8 + j * 8);

            // DENSE: 6 4 2 0 - 7 5 3 1
            uint8x8_t p_1 = xq8p;
            uint8x8_t p_3 = xq8p >> 1;
            uint8x8_t p_5 = xq8p >> 2;
            uint8x8_t p_7 = xq8p >> 3;
            uint8x8_t p_0 = xq8p >> 4;
            uint8x8_t p_2 = xq8p >> 5;
            uint8x8_t p_4 = xq8p >> 6;
            uint8x8_t p_6 = xq8p >> 7;

            uint8x8_t n_1 = xq8n;
            uint8x8_t n_3 = xq8n >> 1;
            uint8x8_t n_5 = xq8n >> 2;
            uint8x8_t n_7 = xq8n >> 3;
            uint8x8_t n_0 = xq8n >> 4;
            uint8x8_t n_2 = xq8n >> 5;
            uint8x8_t n_4 = xq8n >> 6;
            uint8x8_t n_6 = xq8n >> 7;

            // 0/1 masks
            int8x8_t qp0 = vreinterpret_s8_u8(vand_u8(p_0, mask));
            int8x8_t qp1 = vreinterpret_s8_u8(vand_u8(p_1, mask));
            int8x8_t qp2 = vreinterpret_s8_u8(vand_u8(p_2, mask));
            int8x8_t qp3 = vreinterpret_s8_u8(vand_u8(p_3, mask));
            int8x8_t qp4 = vreinterpret_s8_u8(vand_u8(p_4, mask));
            int8x8_t qp5 = vreinterpret_s8_u8(vand_u8(p_5, mask));
            int8x8_t qp6 = vreinterpret_s8_u8(vand_u8(p_6, mask));
            int8x8_t qp7 = vreinterpret_s8_u8(vand_u8(p_7, mask));

            int8x8_t qn0 = vreinterpret_s8_u8(vand_u8(n_0, mask));
            int8x8_t qn1 = vreinterpret_s8_u8(vand_u8(n_1, mask));
            int8x8_t qn2 = vreinterpret_s8_u8(vand_u8(n_2, mask));
            int8x8_t qn3 = vreinterpret_s8_u8(vand_u8(n_3, mask));
            int8x8_t qn4 = vreinterpret_s8_u8(vand_u8(n_4, mask));
            int8x8_t qn5 = vreinterpret_s8_u8(vand_u8(n_5, mask));
            int8x8_t qn6 = vreinterpret_s8_u8(vand_u8(n_6, mask));
            int8x8_t qn7 = vreinterpret_s8_u8(vand_u8(n_7, mask));

            // y blocks
            const int8x16_t y0 = vld1q_s8(y + i * 64 * 16 + j * 64 + 0);
            const int8x16_t y1 = vld1q_s8(y + i * 64 * 16 + j * 64 + 16);
            const int8x16_t y2 = vld1q_s8(y + i * 64 * 16 + j * 64 + 32);
            const int8x16_t y3 = vld1q_s8(y + i * 64 * 16 + j * 64 + 48);

#if defined(__ARM_FEATURE_DOTPROD)
            // Add pos contribution
            int8x16_t mp0 = vcombine_s8(qp0, qp1);
            int8x16_t mp1 = vcombine_s8(qp2, qp3);
            int8x16_t mp2 = vcombine_s8(qp4, qp5);
            int8x16_t mp3 = vcombine_s8(qp6, qp7);

            accu_0 = vdotq_s32(accu_0, y0, mp0);
            accu_1 = vdotq_s32(accu_1, y1, mp1);
            accu_2 = vdotq_s32(accu_2, y2, mp2);
            accu_3 = vdotq_s32(accu_3, y3, mp3);

            // Subtract neg contribution
            int8x16_t mn0 = vcombine_s8(qn0, qn1);
            int8x16_t mn1 = vcombine_s8(qn2, qn3);
            int8x16_t mn2 = vcombine_s8(qn4, qn5);
            int8x16_t mn3 = vcombine_s8(qn6, qn7);

            int32x4_t t0 = vdotq_s32(vdupq_n_s32(0), y0, mn0);
            int32x4_t t1 = vdotq_s32(vdupq_n_s32(0), y1, mn1);
            int32x4_t t2 = vdotq_s32(vdupq_n_s32(0), y2, mn2);
            int32x4_t t3 = vdotq_s32(vdupq_n_s32(0), y3, mn3);
            accu_0 = vsubq_s32(accu_0, t0);
            accu_1 = vsubq_s32(accu_1, t1);
            accu_2 = vsubq_s32(accu_2, t2);
            accu_3 = vsubq_s32(accu_3, t3);
#else
            // Accumulate pos then subtract neg using multiply-sub long
            acc16_0 = vmlal_s8(acc16_0, qp0, vget_low_s8(y0));
            acc16_1 = vmlal_s8(acc16_1, qp1, vget_high_s8(y0));
            acc16_2 = vmlal_s8(acc16_2, qp2, vget_low_s8(y1));
            acc16_3 = vmlal_s8(acc16_3, qp3, vget_high_s8(y1));
            acc16_0 = vmlal_s8(acc16_0, qp4, vget_low_s8(y2));
            acc16_1 = vmlal_s8(acc16_1, qp5, vget_high_s8(y2));
            acc16_2 = vmlal_s8(acc16_2, qp6, vget_low_s8(y3));
            acc16_3 = vmlal_s8(acc16_3, qp7, vget_high_s8(y3));

            acc16_0 = vmlsl_s8(acc16_0, qn0, vget_low_s8(y0));
            acc16_1 = vmlsl_s8(acc16_1, qn1, vget_high_s8(y0));
            acc16_2 = vmlsl_s8(acc16_2, qn2, vget_low_s8(y1));
            acc16_3 = vmlsl_s8(acc16_3, qn3, vget_high_s8(y1));
            acc16_0 = vmlsl_s8(acc16_0, qn4, vget_low_s8(y2));
            acc16_1 = vmlsl_s8(acc16_1, qn5, vget_high_s8(y2));
            acc16_2 = vmlsl_s8(acc16_2, qn6, vget_low_s8(y3));
            acc16_3 = vmlsl_s8(acc16_3, qn7, vget_high_s8(y3));
#endif
        }
#if defined(__ARM_FEATURE_DOTPROD)
        // already in 32-bit accumulators
#else
        accu_0 = vaddq_s32(accu_0, vmovl_s16(vget_low_s16(acc16_0)));
        accu_0 = vaddq_s32(accu_0, vmovl_high_s16(acc16_0));
        accu_1 = vaddq_s32(accu_1, vmovl_s16(vget_low_s16(acc16_1)));
        accu_1 = vaddq_s32(accu_1, vmovl_high_s16(acc16_1));
        accu_2 = vaddq_s32(accu_2, vmovl_s16(vget_low_s16(acc16_2)));
        accu_2 = vaddq_s32(accu_2, vmovl_high_s16(acc16_2));
        accu_3 = vaddq_s32(accu_3, vmovl_s16(vget_low_s16(acc16_3)));
        accu_3 = vaddq_s32(accu_3, vmovl_high_s16(acc16_3));
#endif
    }

    // Leftover blocks
    for (int i = 0; i < groupla_num; ++i) {
#if defined(__ARM_FEATURE_DOTPROD)
#else
        int16x8_t acc16_0 = vdupq_n_s16(0);
        int16x8_t acc16_1 = vdupq_n_s16(0);
        int16x8_t acc16_2 = vdupq_n_s16(0);
        int16x8_t acc16_3 = vdupq_n_s16(0);
#endif
        for (int j = 0; j < la_num; ++j) {
            uint8x8_t xq8p = vld1_u8(xpos + group16_num * 16 * 8 + j * 8);
            uint8x8_t xq8n = vld1_u8(xneg + group16_num * 16 * 8 + j * 8);

            uint8x8_t p_1 = xq8p;
            uint8x8_t p_3 = xq8p >> 1;
            uint8x8_t p_5 = xq8p >> 2;
            uint8x8_t p_7 = xq8p >> 3;
            uint8x8_t p_0 = xq8p >> 4;
            uint8x8_t p_2 = xq8p >> 5;
            uint8x8_t p_4 = xq8p >> 6;
            uint8x8_t p_6 = xq8p >> 7;

            uint8x8_t n_1 = xq8n;
            uint8x8_t n_3 = xq8n >> 1;
            uint8x8_t n_5 = xq8n >> 2;
            uint8x8_t n_7 = xq8n >> 3;
            uint8x8_t n_0 = xq8n >> 4;
            uint8x8_t n_2 = xq8n >> 5;
            uint8x8_t n_4 = xq8n >> 6;
            uint8x8_t n_6 = xq8n >> 7;

            int8x8_t qp0 = vreinterpret_s8_u8(vand_u8(p_0, mask));
            int8x8_t qp1 = vreinterpret_s8_u8(vand_u8(p_1, mask));
            int8x8_t qp2 = vreinterpret_s8_u8(vand_u8(p_2, mask));
            int8x8_t qp3 = vreinterpret_s8_u8(vand_u8(p_3, mask));
            int8x8_t qp4 = vreinterpret_s8_u8(vand_u8(p_4, mask));
            int8x8_t qp5 = vreinterpret_s8_u8(vand_u8(p_5, mask));
            int8x8_t qp6 = vreinterpret_s8_u8(vand_u8(p_6, mask));
            int8x8_t qp7 = vreinterpret_s8_u8(vand_u8(p_7, mask));

            int8x8_t qn0 = vreinterpret_s8_u8(vand_u8(n_0, mask));
            int8x8_t qn1 = vreinterpret_s8_u8(vand_u8(n_1, mask));
            int8x8_t qn2 = vreinterpret_s8_u8(vand_u8(n_2, mask));
            int8x8_t qn3 = vreinterpret_s8_u8(vand_u8(n_3, mask));
            int8x8_t qn4 = vreinterpret_s8_u8(vand_u8(n_4, mask));
            int8x8_t qn5 = vreinterpret_s8_u8(vand_u8(n_5, mask));
            int8x8_t qn6 = vreinterpret_s8_u8(vand_u8(n_6, mask));
            int8x8_t qn7 = vreinterpret_s8_u8(vand_u8(n_7, mask));

            const int base_y = group16_num * 64 * 16 + j * 64;
            const int8x16_t y0 = vld1q_s8(y + base_y + 0);
            const int8x16_t y1 = vld1q_s8(y + base_y + 16);
            const int8x16_t y2 = vld1q_s8(y + base_y + 32);
            const int8x16_t y3 = vld1q_s8(y + base_y + 48);

#if defined(__ARM_FEATURE_DOTPROD)
            int8x16_t mp0 = vcombine_s8(qp0, qp1);
            int8x16_t mp1 = vcombine_s8(qp2, qp3);
            int8x16_t mp2 = vcombine_s8(qp4, qp5);
            int8x16_t mp3 = vcombine_s8(qp6, qp7);
            accu_0 = vdotq_s32(accu_0, y0, mp0);
            accu_1 = vdotq_s32(accu_1, y1, mp1);
            accu_2 = vdotq_s32(accu_2, y2, mp2);
            accu_3 = vdotq_s32(accu_3, y3, mp3);

            int8x16_t mn0 = vcombine_s8(qn0, qn1);
            int8x16_t mn1 = vcombine_s8(qn2, qn3);
            int8x16_t mn2 = vcombine_s8(qn4, qn5);
            int8x16_t mn3 = vcombine_s8(qn6, qn7);
            int32x4_t t0 = vdotq_s32(vdupq_n_s32(0), y0, mn0);
            int32x4_t t1 = vdotq_s32(vdupq_n_s32(0), y1, mn1);
            int32x4_t t2 = vdotq_s32(vdupq_n_s32(0), y2, mn2);
            int32x4_t t3 = vdotq_s32(vdupq_n_s32(0), y3, mn3);
            accu_0 = vsubq_s32(accu_0, t0);
            accu_1 = vsubq_s32(accu_1, t1);
            accu_2 = vsubq_s32(accu_2, t2);
            accu_3 = vsubq_s32(accu_3, t3);
#else
            acc16_0 = vmlal_s8(acc16_0, qp0, vget_low_s8(y0));
            acc16_1 = vmlal_s8(acc16_1, qp1, vget_high_s8(y0));
            acc16_2 = vmlal_s8(acc16_2, qp2, vget_low_s8(y1));
            acc16_3 = vmlal_s8(acc16_3, qp3, vget_high_s8(y1));
            acc16_0 = vmlal_s8(acc16_0, qp4, vget_low_s8(y2));
            acc16_1 = vmlal_s8(acc16_1, qp5, vget_high_s8(y2));
            acc16_2 = vmlal_s8(acc16_2, qp6, vget_low_s8(y3));
            acc16_3 = vmlal_s8(acc16_3, qp7, vget_high_s8(y3));

            acc16_0 = vmlsl_s8(acc16_0, qn0, vget_low_s8(y0));
            acc16_1 = vmlsl_s8(acc16_1, qn1, vget_high_s8(y0));
            acc16_2 = vmlsl_s8(acc16_2, qn2, vget_low_s8(y1));
            acc16_3 = vmlsl_s8(acc16_3, qn3, vget_high_s8(y1));
            acc16_0 = vmlsl_s8(acc16_0, qn4, vget_low_s8(y2));
            acc16_1 = vmlsl_s8(acc16_1, qn5, vget_high_s8(y2));
            acc16_2 = vmlsl_s8(acc16_2, qn6, vget_low_s8(y3));
            acc16_3 = vmlsl_s8(acc16_3, qn7, vget_high_s8(y3));
#endif
        }
#if defined(__ARM_FEATURE_DOTPROD)
#else
        accu_0 = vaddq_s32(accu_0, vmovl_s16(vget_low_s16(acc16_0)));
        accu_0 = vaddq_s32(accu_0, vmovl_high_s16(acc16_0));
        accu_1 = vaddq_s32(accu_1, vmovl_s16(vget_low_s16(acc16_1)));
        accu_1 = vaddq_s32(accu_1, vmovl_high_s16(acc16_1));
        accu_2 = vaddq_s32(accu_2, vmovl_s16(vget_low_s16(acc16_2)));
        accu_2 = vaddq_s32(accu_2, vmovl_high_s16(acc16_2));
        accu_3 = vaddq_s32(accu_3, vmovl_s16(vget_low_s16(acc16_3)));
        accu_3 = vaddq_s32(accu_3, vmovl_high_s16(acc16_3));
#endif
    }

    // Reduce
    accu_0 = vaddq_s32(accu_0, accu_1);
    accu_2 = vaddq_s32(accu_2, accu_3);
    accu_0 = vaddq_s32(accu_0, accu_2);
    int accumi = vaddlvq_s32(accu_0);
    *s = (float)accumi;

#elif defined(__AVX2__)
    __m128i acc32 = _mm_setzero_si128();
    const __m128i zero = _mm_setzero_si128();
    const __m128i one16 = _mm_set1_epi16(1);

    for (int i = 0; i < group16_num; ++i) {
        for (int j = 0; j < 16; ++j) {
            const uint8_t *xbp = xpos + i * 16 * 8 + j * 8;
            const uint8_t *xbn = xneg + i * 16 * 8 + j * 8;
            __m128i xp8 = _mm_loadl_epi64((const __m128i *)xbp);
            __m128i xn8 = _mm_loadl_epi64((const __m128i *)xbn);
            __m128i xp16 = _mm_unpacklo_epi8(xp8, zero);
            __m128i xn16 = _mm_unpacklo_epi8(xn8, zero);

            // Build masks for pos
            __m128i bp1 = _mm_and_si128(_mm_srli_epi16(xp16, 0), one16);
            __m128i bp3 = _mm_and_si128(_mm_srli_epi16(xp16, 1), one16);
            __m128i bp5 = _mm_and_si128(_mm_srli_epi16(xp16, 2), one16);
            __m128i bp7 = _mm_and_si128(_mm_srli_epi16(xp16, 3), one16);
            __m128i bp0 = _mm_and_si128(_mm_srli_epi16(xp16, 4), one16);
            __m128i bp2 = _mm_and_si128(_mm_srli_epi16(xp16, 5), one16);
            __m128i bp4 = _mm_and_si128(_mm_srli_epi16(xp16, 6), one16);
            __m128i bp6 = _mm_and_si128(_mm_srli_epi16(xp16, 7), one16);

            __m128i mp0 = _mm_packs_epi16(bp0, bp1);
            __m128i mp1 = _mm_packs_epi16(bp2, bp3);
            __m128i mp2 = _mm_packs_epi16(bp4, bp5);
            __m128i mp3 = _mm_packs_epi16(bp6, bp7);

            // Build masks for neg
            __m128i bn1 = _mm_and_si128(_mm_srli_epi16(xn16, 0), one16);
            __m128i bn3 = _mm_and_si128(_mm_srli_epi16(xn16, 1), one16);
            __m128i bn5 = _mm_and_si128(_mm_srli_epi16(xn16, 2), one16);
            __m128i bn7 = _mm_and_si128(_mm_srli_epi16(xn16, 3), one16);
            __m128i bn0 = _mm_and_si128(_mm_srli_epi16(xn16, 4), one16);
            __m128i bn2 = _mm_and_si128(_mm_srli_epi16(xn16, 5), one16);
            __m128i bn4 = _mm_and_si128(_mm_srli_epi16(xn16, 6), one16);
            __m128i bn6 = _mm_and_si128(_mm_srli_epi16(xn16, 7), one16);

            __m128i mn0 = _mm_packs_epi16(bn0, bn1);
            __m128i mn1 = _mm_packs_epi16(bn2, bn3);
            __m128i mn2 = _mm_packs_epi16(bn4, bn5);
            __m128i mn3 = _mm_packs_epi16(bn6, bn7);

            const int base_y = i * 64 * 16 + j * 64;
            __m128i y0 = _mm_loadu_si128((const __m128i *)(y + base_y + 0));
            __m128i y1 = _mm_loadu_si128((const __m128i *)(y + base_y + 16));
            __m128i y2 = _mm_loadu_si128((const __m128i *)(y + base_y + 32));
            __m128i y3 = _mm_loadu_si128((const __m128i *)(y + base_y + 48));

            // pos - neg per block
            __m128i pp0 = _mm_maddubs_epi16(mp0, y0);
            __m128i pn0 = _mm_maddubs_epi16(mn0, y0);
            __m128i pd0 = _mm_sub_epi16(pp0, pn0);
            acc32 = _mm_add_epi32(acc32, _mm_madd_epi16(pd0, one16));

            __m128i pp1 = _mm_maddubs_epi16(mp1, y1);
            __m128i pn1 = _mm_maddubs_epi16(mn1, y1);
            __m128i pd1 = _mm_sub_epi16(pp1, pn1);
            acc32 = _mm_add_epi32(acc32, _mm_madd_epi16(pd1, one16));

            __m128i pp2 = _mm_maddubs_epi16(mp2, y2);
            __m128i pn2 = _mm_maddubs_epi16(mn2, y2);
            __m128i pd2 = _mm_sub_epi16(pp2, pn2);
            acc32 = _mm_add_epi32(acc32, _mm_madd_epi16(pd2, one16));

            __m128i pp3 = _mm_maddubs_epi16(mp3, y3);
            __m128i pn3 = _mm_maddubs_epi16(mn3, y3);
            __m128i pd3 = _mm_sub_epi16(pp3, pn3);
            acc32 = _mm_add_epi32(acc32, _mm_madd_epi16(pd3, one16));
        }
    }

    for (int i = 0; i < groupla_num; ++i) {
        for (int j = 0; j < la_num; ++j) {
            const uint8_t *xbp = xpos + group16_num * 16 * 8 + j * 8;
            const uint8_t *xbn = xneg + group16_num * 16 * 8 + j * 8;
            __m128i xp8 = _mm_loadl_epi64((const __m128i *)xbp);
            __m128i xn8 = _mm_loadl_epi64((const __m128i *)xbn);
            __m128i xp16 = _mm_unpacklo_epi8(xp8, zero);
            __m128i xn16 = _mm_unpacklo_epi8(xn8, zero);

            __m128i bp1 = _mm_and_si128(_mm_srli_epi16(xp16, 0), one16);
            __m128i bp3 = _mm_and_si128(_mm_srli_epi16(xp16, 1), one16);
            __m128i bp5 = _mm_and_si128(_mm_srli_epi16(xp16, 2), one16);
            __m128i bp7 = _mm_and_si128(_mm_srli_epi16(xp16, 3), one16);
            __m128i bp0 = _mm_and_si128(_mm_srli_epi16(xp16, 4), one16);
            __m128i bp2 = _mm_and_si128(_mm_srli_epi16(xp16, 5), one16);
            __m128i bp4 = _mm_and_si128(_mm_srli_epi16(xp16, 6), one16);
            __m128i bp6 = _mm_and_si128(_mm_srli_epi16(xp16, 7), one16);

            __m128i mp0 = _mm_packs_epi16(bp0, bp1);
            __m128i mp1 = _mm_packs_epi16(bp2, bp3);
            __m128i mp2 = _mm_packs_epi16(bp4, bp5);
            __m128i mp3 = _mm_packs_epi16(bp6, bp7);

            __m128i bn1 = _mm_and_si128(_mm_srli_epi16(xn16, 0), one16);
            __m128i bn3 = _mm_and_si128(_mm_srli_epi16(xn16, 1), one16);
            __m128i bn5 = _mm_and_si128(_mm_srli_epi16(xn16, 2), one16);
            __m128i bn7 = _mm_and_si128(_mm_srli_epi16(xn16, 3), one16);
            __m128i bn0 = _mm_and_si128(_mm_srli_epi16(xn16, 4), one16);
            __m128i bn2 = _mm_and_si128(_mm_srli_epi16(xn16, 5), one16);
            __m128i bn4 = _mm_and_si128(_mm_srli_epi16(xn16, 6), one16);
            __m128i bn6 = _mm_and_si128(_mm_srli_epi16(xn16, 7), one16);

            __m128i mn0 = _mm_packs_epi16(bn0, bn1);
            __m128i mn1 = _mm_packs_epi16(bn2, bn3);
            __m128i mn2 = _mm_packs_epi16(bn4, bn5);
            __m128i mn3 = _mm_packs_epi16(bn6, bn7);

            const int base_y = group16_num * 64 * 16 + j * 64;
            __m128i y0 = _mm_loadu_si128((const __m128i *)(y + base_y + 0));
            __m128i y1 = _mm_loadu_si128((const __m128i *)(y + base_y + 16));
            __m128i y2 = _mm_loadu_si128((const __m128i *)(y + base_y + 32));
            __m128i y3 = _mm_loadu_si128((const __m128i *)(y + base_y + 48));

            __m128i pp0 = _mm_maddubs_epi16(mp0, y0);
            __m128i pn0 = _mm_maddubs_epi16(mn0, y0);
            __m128i pd0 = _mm_sub_epi16(pp0, pn0);
            acc32 = _mm_add_epi32(acc32, _mm_madd_epi16(pd0, one16));

            __m128i pp1 = _mm_maddubs_epi16(mp1, y1);
            __m128i pn1 = _mm_maddubs_epi16(mn1, y1);
            __m128i pd1 = _mm_sub_epi16(pp1, pn1);
            acc32 = _mm_add_epi32(acc32, _mm_madd_epi16(pd1, one16));

            __m128i pp2 = _mm_maddubs_epi16(mp2, y2);
            __m128i pn2 = _mm_maddubs_epi16(mn2, y2);
            __m128i pd2 = _mm_sub_epi16(pp2, pn2);
            acc32 = _mm_add_epi32(acc32, _mm_madd_epi16(pd2, one16));

            __m128i pp3 = _mm_maddubs_epi16(mp3, y3);
            __m128i pn3 = _mm_maddubs_epi16(mn3, y3);
            __m128i pd3 = _mm_sub_epi16(pp3, pn3);
            acc32 = _mm_add_epi32(acc32, _mm_madd_epi16(pd3, one16));
        }
    }

    __m128i shuf = _mm_shuffle_epi32(acc32, _MM_SHUFFLE(2, 3, 0, 1));
    acc32 = _mm_add_epi32(acc32, shuf);
    shuf = _mm_shuffle_epi32(acc32, _MM_SHUFFLE(1, 0, 3, 2));
    acc32 = _mm_add_epi32(acc32, shuf);
    int accumi = _mm_cvtsi128_si32(acc32);
    *s = (float)accumi;

#else
    // Portable fallback
    int sum = 0;
    for (int c = 0; c < nb; ++c) {
        const uint8_t *bp = xpos + (size_t)c * 8;
        const uint8_t *bn = xneg + (size_t)c * 8;
        const int8_t *yy = y + (size_t)c * 64;
        for (int sidx = 0; sidx < 64; ++sidx) {
            int g = sidx / 16;      // 0..3
            int h = (sidx / 8) & 1; // 0 or 1
            int out_bit = h ? g : (4 + g);
            int out_byte = sidx & 7;
            uint8_t selp = (uint8_t)((bp[out_byte] >> out_bit) & 1u);
            uint8_t seln = (uint8_t)((bn[out_byte] >> out_bit) & 1u);
            sum += ((int)selp - (int)seln) * (int)yy[sidx];
        }
    }
    *s = (float)sum;
#endif
}

static void preprocess_weights(const int &nb,
                               const uint8_t *&quant_weight_ptr,
                               uint8_t *array_buffer,
                               uint8_t *binary_buffer_0,
                               uint8_t *binary_buffer_1) {
    // Stage original 2-bit data (memcpy 32B per 128 weights)
    // Split ternary into two binary matrices and pack 64 weights -> 8 bytes

    const int32_t num_i2_blocks = nb;   // number of 128-weight (32B) blocks
    const int32_t bytes_per_block = 32; // 32 bytes per 128 weights

    // Clear destination buffers
    memset(array_buffer, 0, (size_t)num_i2_blocks * bytes_per_block);
    memset(binary_buffer_0, 0, (size_t)num_i2_blocks * 2 * 8);
    memset(binary_buffer_1, 0, (size_t)num_i2_blocks * 2 * 8);

#if defined(__ARM_NEON)
    const uint8x8_t bit_0 = vdup_n_u8(1u << 0);
    const uint8x8_t bit_1 = vdup_n_u8(1u << 1);
    const uint8x8_t bit_2 = vdup_n_u8(1u << 2);
    const uint8x8_t bit_3 = vdup_n_u8(1u << 3);
    const uint8x8_t bit_4 = vdup_n_u8(1u << 4);
    const uint8x8_t bit_5 = vdup_n_u8(1u << 5);
    const uint8x8_t bit_6 = vdup_n_u8(1u << 6);
    const uint8x8_t bit_7 = vdup_n_u8(1u << 7);

    const int8x16_t two16 = vdupq_n_s8(2);
    const int8x16_t zero16 = vdupq_n_s8(0);
    const uint8x16_t mask_vec = vdupq_n_u8(3);
#elif defined(__AVX2__)
    const __m128i bit_0 = _mm_set1_epi8((char)(1u << 0));
    const __m128i bit_1 = _mm_set1_epi8((char)(1u << 1));
    const __m128i bit_2 = _mm_set1_epi8((char)(1u << 2));
    const __m128i bit_3 = _mm_set1_epi8((char)(1u << 3));
    const __m128i bit_4 = _mm_set1_epi8((char)(1u << 4));
    const __m128i bit_5 = _mm_set1_epi8((char)(1u << 5));
    const __m128i bit_6 = _mm_set1_epi8((char)(1u << 6));
    const __m128i bit_7 = _mm_set1_epi8((char)(1u << 7));
    const __m128i two16 = _mm_set1_epi16(2);
    const __m128i zero16 = _mm_set1_epi16(0);
    const __m128i mask16 = _mm_set1_epi16(3);
#endif

    for (int32_t b = 0; b < num_i2_blocks; ++b) {
        const uint8_t *src = quant_weight_ptr + (size_t)b * bytes_per_block;
        __builtin_prefetch(src + 64, 0, 1);
        uint8_t *dst = array_buffer + (size_t)b * bytes_per_block;

        // Load 32B (128 weights) and stage to array_buffer
#if defined(__ARM_NEON)
        uint8x16_t xq8_lo = vld1q_u8(src);
        uint8x16_t xq8_hi = vld1q_u8(src + 16);

        vst1q_u8(dst, xq8_lo);
        vst1q_u8(dst + 16, xq8_hi);

        // Extract 2-bit fields (groups 0..3) for pos 0..15 and 16..31
        uint8x16_t sh2_lo = vshrq_n_u8(xq8_lo, 2);
        uint8x16_t sh2_hi = vshrq_n_u8(xq8_hi, 2);
        uint8x16_t sh4_lo = vshrq_n_u8(xq8_lo, 4);
        uint8x16_t sh4_hi = vshrq_n_u8(xq8_hi, 4);
        uint8x16_t sh6_lo = vshrq_n_u8(xq8_lo, 6);
        uint8x16_t sh6_hi = vshrq_n_u8(xq8_hi, 6);

        int8x16_t q0 = vreinterpretq_s8_u8(vandq_u8(sh6_lo, mask_vec)); // group 0, pos 0..15
        int8x16_t q1 = vreinterpretq_s8_u8(vandq_u8(sh6_hi, mask_vec)); // group 0, pos 16..31
        int8x16_t q2 = vreinterpretq_s8_u8(vandq_u8(sh4_lo, mask_vec)); // group 1, pos 0..15
        int8x16_t q3 = vreinterpretq_s8_u8(vandq_u8(sh4_hi, mask_vec)); // group 1, pos 16..31
        int8x16_t q4 = vreinterpretq_s8_u8(vandq_u8(sh2_lo, mask_vec)); // group 2, pos 0..15
        int8x16_t q5 = vreinterpretq_s8_u8(vandq_u8(sh2_hi, mask_vec)); // group 2, pos 16..31
        int8x16_t q6 = vreinterpretq_s8_u8(vandq_u8(xq8_lo, mask_vec)); // group 3, pos 0..15
        int8x16_t q7 = vreinterpretq_s8_u8(vandq_u8(xq8_hi, mask_vec)); // group 3, pos 16..31
#elif defined(__AVX2__)
        __m128i xq8_lo = _mm_loadu_si128((const __m128i *)src);
        __m128i xq8_hi = _mm_loadu_si128((const __m128i *)(src + 16));
        _mm_storeu_si128((__m128i *)dst, xq8_lo);
        _mm_storeu_si128((__m128i *)(dst + 16), xq8_hi);

        const __m128i zero = _mm_setzero_si128();
        __m128i xlo_lo = _mm_unpacklo_epi8(xq8_lo, zero);
        __m128i xlo_hi = _mm_unpackhi_epi8(xq8_lo, zero);
        __m128i xhi_lo = _mm_unpacklo_epi8(xq8_hi, zero);
        __m128i xhi_hi = _mm_unpackhi_epi8(xq8_hi, zero);

        __m128i g0_lo = _mm_and_si128(_mm_srli_epi16(xlo_lo, 6), mask16);
        __m128i g0_hi = _mm_and_si128(_mm_srli_epi16(xlo_hi, 6), mask16);
        __m128i g0_l2 = _mm_and_si128(_mm_srli_epi16(xhi_lo, 6), mask16);
        __m128i g0_h2 = _mm_and_si128(_mm_srli_epi16(xhi_hi, 6), mask16);

        __m128i g1_lo = _mm_and_si128(_mm_srli_epi16(xlo_lo, 4), mask16);
        __m128i g1_hi = _mm_and_si128(_mm_srli_epi16(xlo_hi, 4), mask16);
        __m128i g1_l2 = _mm_and_si128(_mm_srli_epi16(xhi_lo, 4), mask16);
        __m128i g1_h2 = _mm_and_si128(_mm_srli_epi16(xhi_hi, 4), mask16);

        __m128i g2_lo = _mm_and_si128(_mm_srli_epi16(xlo_lo, 2), mask16);
        __m128i g2_hi = _mm_and_si128(_mm_srli_epi16(xlo_hi, 2), mask16);
        __m128i g2_l2 = _mm_and_si128(_mm_srli_epi16(xhi_lo, 2), mask16);
        __m128i g2_h2 = _mm_and_si128(_mm_srli_epi16(xhi_hi, 2), mask16);

        __m128i g3_lo = _mm_and_si128(xlo_lo, mask16);
        __m128i g3_hi = _mm_and_si128(xlo_hi, mask16);
        __m128i g3_l2 = _mm_and_si128(xhi_lo, mask16);
        __m128i g3_h2 = _mm_and_si128(xhi_hi, mask16);
#else
        // Scalar fallback: stage and pack using portable ops
        memcpy(dst, src, (size_t)bytes_per_block);
        for (int half = 0; half < 2; ++half) {
            uint8_t pack_b1[8] = {0};
            uint8_t pack_b2[8] = {0};
            for (int s = 0; s < 64; ++s) {
                int j = half * 64 + s;
                int group = j / 32;
                int pos = j % 32;
                uint8_t byte = src[pos];
                uint8_t q = (uint8_t)((byte >> (6 - 2 * group)) & 0x03);
                uint8_t is_pos = (uint8_t)(q == 2);
                uint8_t is_neg = (uint8_t)(q == 0);
                int g = s / 16;
                int h = (s / 8) & 1;
                int out_bit = h ? g : (4 + g);
                int out_byte = s & 7;
                pack_b1[out_byte] |= (uint8_t)(is_pos << out_bit);
                pack_b2[out_byte] |= (uint8_t)(is_neg << out_bit);
            }
            int chunk_idx = b * 2 + half;
            memcpy(binary_buffer_0 + (size_t)chunk_idx * 8, pack_b1, 8);
            memcpy(binary_buffer_1 + (size_t)chunk_idx * 8, pack_b2, 8);
        }
        continue;
#endif

        // Compute pos/neg masks for all 16 lanes at once
#if defined(__ARM_NEON)
        uint8x16_t p0q = vreinterpretq_u8_s8(vceqq_s8(q0, two16));
        uint8x16_t n0q = vreinterpretq_u8_s8(vceqq_s8(q0, zero16));
        uint8x16_t p1q = vreinterpretq_u8_s8(vceqq_s8(q1, two16));
        uint8x16_t n1q = vreinterpretq_u8_s8(vceqq_s8(q1, zero16));
        uint8x16_t p2q = vreinterpretq_u8_s8(vceqq_s8(q2, two16));
        uint8x16_t n2q = vreinterpretq_u8_s8(vceqq_s8(q2, zero16));
        uint8x16_t p3q = vreinterpretq_u8_s8(vceqq_s8(q3, two16));
        uint8x16_t n3q = vreinterpretq_u8_s8(vceqq_s8(q3, zero16));
        uint8x16_t p4q = vreinterpretq_u8_s8(vceqq_s8(q4, two16));
        uint8x16_t n4q = vreinterpretq_u8_s8(vceqq_s8(q4, zero16));
        uint8x16_t p5q = vreinterpretq_u8_s8(vceqq_s8(q5, two16));
        uint8x16_t n5q = vreinterpretq_u8_s8(vceqq_s8(q5, zero16));
        uint8x16_t p6q = vreinterpretq_u8_s8(vceqq_s8(q6, two16));
        uint8x16_t n6q = vreinterpretq_u8_s8(vceqq_s8(q6, zero16));
        uint8x16_t p7q = vreinterpretq_u8_s8(vceqq_s8(q7, two16));
        uint8x16_t n7q = vreinterpretq_u8_s8(vceqq_s8(q7, zero16));
#elif defined(__AVX2__)
        __m128i p0_lo = _mm_cmpeq_epi16(g0_lo, two16);
        __m128i p0_hi = _mm_cmpeq_epi16(g0_hi, two16);
        __m128i p1_lo = _mm_cmpeq_epi16(g0_l2, two16);
        __m128i p1_hi = _mm_cmpeq_epi16(g0_h2, two16);
        __m128i p2_lo = _mm_cmpeq_epi16(g1_lo, two16);
        __m128i p2_hi = _mm_cmpeq_epi16(g1_hi, two16);
        __m128i p3_lo = _mm_cmpeq_epi16(g1_l2, two16);
        __m128i p3_hi = _mm_cmpeq_epi16(g1_h2, two16);
        __m128i p4_lo = _mm_cmpeq_epi16(g2_lo, two16);
        __m128i p4_hi = _mm_cmpeq_epi16(g2_hi, two16);
        __m128i p5_lo = _mm_cmpeq_epi16(g2_l2, two16);
        __m128i p5_hi = _mm_cmpeq_epi16(g2_h2, two16);
        __m128i p6_lo = _mm_cmpeq_epi16(g3_lo, two16);
        __m128i p6_hi = _mm_cmpeq_epi16(g3_hi, two16);
        __m128i p7_lo = _mm_cmpeq_epi16(g3_l2, two16);
        __m128i p7_hi = _mm_cmpeq_epi16(g3_h2, two16);

        __m128i n0_lo = _mm_cmpeq_epi16(g0_lo, zero16);
        __m128i n0_hi = _mm_cmpeq_epi16(g0_hi, zero16);
        __m128i n1_lo = _mm_cmpeq_epi16(g0_l2, zero16);
        __m128i n1_hi = _mm_cmpeq_epi16(g0_h2, zero16);
        __m128i n2_lo = _mm_cmpeq_epi16(g1_lo, zero16);
        __m128i n2_hi = _mm_cmpeq_epi16(g1_hi, zero16);
        __m128i n3_lo = _mm_cmpeq_epi16(g1_l2, zero16);
        __m128i n3_hi = _mm_cmpeq_epi16(g1_h2, zero16);
        __m128i n4_lo = _mm_cmpeq_epi16(g2_lo, zero16);
        __m128i n4_hi = _mm_cmpeq_epi16(g2_hi, zero16);
        __m128i n5_lo = _mm_cmpeq_epi16(g2_l2, zero16);
        __m128i n5_hi = _mm_cmpeq_epi16(g2_h2, zero16);
        __m128i n6_lo = _mm_cmpeq_epi16(g3_lo, zero16);
        __m128i n6_hi = _mm_cmpeq_epi16(g3_hi, zero16);
        __m128i n7_lo = _mm_cmpeq_epi16(g3_l2, zero16);
        __m128i n7_hi = _mm_cmpeq_epi16(g3_h2, zero16);

        // Narrow to 8-bit masks (0xFF or 0x00) for 8 lanes
        __m128i zero16x = _mm_setzero_si128();
        // Use signed pack to convert 0xFFFF -> 0xFF, 0x0000 -> 0x00
        __m128i p0_lo8 = _mm_packs_epi16(p0_lo, zero16x);
        __m128i p0_hi8 = _mm_packs_epi16(p0_hi, zero16x);
        __m128i p1_lo8 = _mm_packs_epi16(p1_lo, zero16x);
        __m128i p1_hi8 = _mm_packs_epi16(p1_hi, zero16x);
        __m128i p2_lo8 = _mm_packs_epi16(p2_lo, zero16x);
        __m128i p2_hi8 = _mm_packs_epi16(p2_hi, zero16x);
        __m128i p3_lo8 = _mm_packs_epi16(p3_lo, zero16x);
        __m128i p3_hi8 = _mm_packs_epi16(p3_hi, zero16x);
        __m128i p4_lo8 = _mm_packs_epi16(p4_lo, zero16x);
        __m128i p4_hi8 = _mm_packs_epi16(p4_hi, zero16x);
        __m128i p5_lo8 = _mm_packs_epi16(p5_lo, zero16x);
        __m128i p5_hi8 = _mm_packs_epi16(p5_hi, zero16x);
        __m128i p6_lo8 = _mm_packs_epi16(p6_lo, zero16x);
        __m128i p6_hi8 = _mm_packs_epi16(p6_hi, zero16x);
        __m128i p7_lo8 = _mm_packs_epi16(p7_lo, zero16x);
        __m128i p7_hi8 = _mm_packs_epi16(p7_hi, zero16x);

        __m128i n0_lo8 = _mm_packs_epi16(n0_lo, zero16x);
        __m128i n0_hi8 = _mm_packs_epi16(n0_hi, zero16x);
        __m128i n1_lo8 = _mm_packs_epi16(n1_lo, zero16x);
        __m128i n1_hi8 = _mm_packs_epi16(n1_hi, zero16x);
        __m128i n2_lo8 = _mm_packs_epi16(n2_lo, zero16x);
        __m128i n2_hi8 = _mm_packs_epi16(n2_hi, zero16x);
        __m128i n3_lo8 = _mm_packs_epi16(n3_lo, zero16x);
        __m128i n3_hi8 = _mm_packs_epi16(n3_hi, zero16x);
        __m128i n4_lo8 = _mm_packs_epi16(n4_lo, zero16x);
        __m128i n4_hi8 = _mm_packs_epi16(n4_hi, zero16x);
        __m128i n5_lo8 = _mm_packs_epi16(n5_lo, zero16x);
        __m128i n5_hi8 = _mm_packs_epi16(n5_hi, zero16x);
        __m128i n6_lo8 = _mm_packs_epi16(n6_lo, zero16x);
        __m128i n6_hi8 = _mm_packs_epi16(n6_hi, zero16x);
        __m128i n7_lo8 = _mm_packs_epi16(n7_lo, zero16x);
        __m128i n7_hi8 = _mm_packs_epi16(n7_hi, zero16x);
#endif

        // Half 0 (j = 0..63)
#if defined(__ARM_NEON)
        uint8x8_t pack_p_h0 = vdup_n_u8(0);
        uint8x8_t pack_n_h0 = vdup_n_u8(0);

        uint8x8_t p0_lo8 = vget_low_u8(p0q);
        uint8x8_t p0_hi8 = vget_high_u8(p0q);
        uint8x8_t n0_lo8 = vget_low_u8(n0q);
        uint8x8_t n0_hi8 = vget_high_u8(n0q);
        uint8x8_t p1_lo8 = vget_low_u8(p1q);
        uint8x8_t p1_hi8 = vget_high_u8(p1q);
        uint8x8_t n1_lo8 = vget_low_u8(n1q);
        uint8x8_t n1_hi8 = vget_high_u8(n1q);
        uint8x8_t p2_lo8 = vget_low_u8(p2q);
        uint8x8_t p2_hi8 = vget_high_u8(p2q);
        uint8x8_t n2_lo8 = vget_low_u8(n2q);
        uint8x8_t n2_hi8 = vget_high_u8(n2q);
        uint8x8_t p3_lo8 = vget_low_u8(p3q);
        uint8x8_t p3_hi8 = vget_high_u8(p3q);
        uint8x8_t n3_lo8 = vget_low_u8(n3q);
        uint8x8_t n3_hi8 = vget_high_u8(n3q);

        // Positive
        pack_p_h0 = vorr_u8(pack_p_h0, vand_u8(p0_lo8, bit_4));
        pack_p_h0 = vorr_u8(pack_p_h0, vand_u8(p0_hi8, bit_0));
        pack_p_h0 = vorr_u8(pack_p_h0, vand_u8(p1_lo8, bit_5));
        pack_p_h0 = vorr_u8(pack_p_h0, vand_u8(p1_hi8, bit_1));
        pack_p_h0 = vorr_u8(pack_p_h0, vand_u8(p2_lo8, bit_6));
        pack_p_h0 = vorr_u8(pack_p_h0, vand_u8(p2_hi8, bit_2));
        pack_p_h0 = vorr_u8(pack_p_h0, vand_u8(p3_lo8, bit_7));
        pack_p_h0 = vorr_u8(pack_p_h0, vand_u8(p3_hi8, bit_3));

        // Negative
        pack_n_h0 = vorr_u8(pack_n_h0, vand_u8(n0_lo8, bit_4));
        pack_n_h0 = vorr_u8(pack_n_h0, vand_u8(n0_hi8, bit_0));
        pack_n_h0 = vorr_u8(pack_n_h0, vand_u8(n1_lo8, bit_5));
        pack_n_h0 = vorr_u8(pack_n_h0, vand_u8(n1_hi8, bit_1));
        pack_n_h0 = vorr_u8(pack_n_h0, vand_u8(n2_lo8, bit_6));
        pack_n_h0 = vorr_u8(pack_n_h0, vand_u8(n2_hi8, bit_2));
        pack_n_h0 = vorr_u8(pack_n_h0, vand_u8(n3_lo8, bit_7));
        pack_n_h0 = vorr_u8(pack_n_h0, vand_u8(n3_hi8, bit_3));
#elif defined(__AVX2__)
        __m128i pack_p_h0 = _mm_setzero_si128();
        __m128i pack_n_h0 = _mm_setzero_si128();
        pack_p_h0 = _mm_or_si128(pack_p_h0, _mm_and_si128(p0_lo8, bit_4));
        pack_p_h0 = _mm_or_si128(pack_p_h0, _mm_and_si128(p0_hi8, bit_0));
        pack_p_h0 = _mm_or_si128(pack_p_h0, _mm_and_si128(p1_lo8, bit_5));
        pack_p_h0 = _mm_or_si128(pack_p_h0, _mm_and_si128(p1_hi8, bit_1));
        pack_p_h0 = _mm_or_si128(pack_p_h0, _mm_and_si128(p2_lo8, bit_6));
        pack_p_h0 = _mm_or_si128(pack_p_h0, _mm_and_si128(p2_hi8, bit_2));
        pack_p_h0 = _mm_or_si128(pack_p_h0, _mm_and_si128(p3_lo8, bit_7));
        pack_p_h0 = _mm_or_si128(pack_p_h0, _mm_and_si128(p3_hi8, bit_3));

        pack_n_h0 = _mm_or_si128(pack_n_h0, _mm_and_si128(n0_lo8, bit_4));
        pack_n_h0 = _mm_or_si128(pack_n_h0, _mm_and_si128(n0_hi8, bit_0));
        pack_n_h0 = _mm_or_si128(pack_n_h0, _mm_and_si128(n1_lo8, bit_5));
        pack_n_h0 = _mm_or_si128(pack_n_h0, _mm_and_si128(n1_hi8, bit_1));
        pack_n_h0 = _mm_or_si128(pack_n_h0, _mm_and_si128(n2_lo8, bit_6));
        pack_n_h0 = _mm_or_si128(pack_n_h0, _mm_and_si128(n2_hi8, bit_2));
        pack_n_h0 = _mm_or_si128(pack_n_h0, _mm_and_si128(n3_lo8, bit_7));
        pack_n_h0 = _mm_or_si128(pack_n_h0, _mm_and_si128(n3_hi8, bit_3));
#endif

        // Half 1 (j = 64..127)
#if defined(__ARM_NEON)
        uint8x8_t pack_p_h1 = vdup_n_u8(0);
        uint8x8_t pack_n_h1 = vdup_n_u8(0);

        uint8x8_t p4_lo8 = vget_low_u8(p4q);
        uint8x8_t p4_hi8 = vget_high_u8(p4q);
        uint8x8_t n4_lo8 = vget_low_u8(n4q);
        uint8x8_t n4_hi8 = vget_high_u8(n4q);

        uint8x8_t p5_lo8 = vget_low_u8(p5q);
        uint8x8_t p5_hi8 = vget_high_u8(p5q);
        uint8x8_t n5_lo8 = vget_low_u8(n5q);
        uint8x8_t n5_hi8 = vget_high_u8(n5q);

        uint8x8_t p6_lo8 = vget_low_u8(p6q);
        uint8x8_t p6_hi8 = vget_high_u8(p6q);
        uint8x8_t n6_lo8 = vget_low_u8(n6q);
        uint8x8_t n6_hi8 = vget_high_u8(n6q);

        uint8x8_t p7_lo8 = vget_low_u8(p7q);
        uint8x8_t p7_hi8 = vget_high_u8(p7q);
        uint8x8_t n7_lo8 = vget_low_u8(n7q);
        uint8x8_t n7_hi8 = vget_high_u8(n7q);

        // Positive
        pack_p_h1 = vorr_u8(pack_p_h1, vand_u8(p4_lo8, bit_4));
        pack_p_h1 = vorr_u8(pack_p_h1, vand_u8(p4_hi8, bit_0));
        pack_p_h1 = vorr_u8(pack_p_h1, vand_u8(p5_lo8, bit_5));
        pack_p_h1 = vorr_u8(pack_p_h1, vand_u8(p5_hi8, bit_1));
        pack_p_h1 = vorr_u8(pack_p_h1, vand_u8(p6_lo8, bit_6));
        pack_p_h1 = vorr_u8(pack_p_h1, vand_u8(p6_hi8, bit_2));
        pack_p_h1 = vorr_u8(pack_p_h1, vand_u8(p7_lo8, bit_7));
        pack_p_h1 = vorr_u8(pack_p_h1, vand_u8(p7_hi8, bit_3));

        // Negative
        pack_n_h1 = vorr_u8(pack_n_h1, vand_u8(n4_lo8, bit_4));
        pack_n_h1 = vorr_u8(pack_n_h1, vand_u8(n4_hi8, bit_0));
        pack_n_h1 = vorr_u8(pack_n_h1, vand_u8(n5_lo8, bit_5));
        pack_n_h1 = vorr_u8(pack_n_h1, vand_u8(n5_hi8, bit_1));
        pack_n_h1 = vorr_u8(pack_n_h1, vand_u8(n6_lo8, bit_6));
        pack_n_h1 = vorr_u8(pack_n_h1, vand_u8(n6_hi8, bit_2));
        pack_n_h1 = vorr_u8(pack_n_h1, vand_u8(n7_lo8, bit_7));
        pack_n_h1 = vorr_u8(pack_n_h1, vand_u8(n7_hi8, bit_3));

        // Store results
        int32_t chunk_idx0 = b * 2 + 0;
        int32_t chunk_idx1 = b * 2 + 1;
        vst1_u8(binary_buffer_0 + (size_t)chunk_idx0 * 8, pack_p_h0);
        vst1_u8(binary_buffer_1 + (size_t)chunk_idx0 * 8, pack_n_h0);
        vst1_u8(binary_buffer_0 + (size_t)chunk_idx1 * 8, pack_p_h1);
        vst1_u8(binary_buffer_1 + (size_t)chunk_idx1 * 8, pack_n_h1);
#elif defined(__AVX2__)
        __m128i pack_p_h1 = _mm_setzero_si128();
        __m128i pack_n_h1 = _mm_setzero_si128();

        pack_p_h1 = _mm_or_si128(pack_p_h1, _mm_and_si128(p4_lo8, bit_4));
        pack_p_h1 = _mm_or_si128(pack_p_h1, _mm_and_si128(p4_hi8, bit_0));
        pack_p_h1 = _mm_or_si128(pack_p_h1, _mm_and_si128(p5_lo8, bit_5));
        pack_p_h1 = _mm_or_si128(pack_p_h1, _mm_and_si128(p5_hi8, bit_1));
        pack_p_h1 = _mm_or_si128(pack_p_h1, _mm_and_si128(p6_lo8, bit_6));
        pack_p_h1 = _mm_or_si128(pack_p_h1, _mm_and_si128(p6_hi8, bit_2));
        pack_p_h1 = _mm_or_si128(pack_p_h1, _mm_and_si128(p7_lo8, bit_7));
        pack_p_h1 = _mm_or_si128(pack_p_h1, _mm_and_si128(p7_hi8, bit_3));

        pack_n_h1 = _mm_or_si128(pack_n_h1, _mm_and_si128(n4_lo8, bit_4));
        pack_n_h1 = _mm_or_si128(pack_n_h1, _mm_and_si128(n4_hi8, bit_0));
        pack_n_h1 = _mm_or_si128(pack_n_h1, _mm_and_si128(n5_lo8, bit_5));
        pack_n_h1 = _mm_or_si128(pack_n_h1, _mm_and_si128(n5_hi8, bit_1));
        pack_n_h1 = _mm_or_si128(pack_n_h1, _mm_and_si128(n6_lo8, bit_6));
        pack_n_h1 = _mm_or_si128(pack_n_h1, _mm_and_si128(n6_hi8, bit_2));
        pack_n_h1 = _mm_or_si128(pack_n_h1, _mm_and_si128(n7_lo8, bit_7));
        pack_n_h1 = _mm_or_si128(pack_n_h1, _mm_and_si128(n7_hi8, bit_3));

        // Store results
        int32_t chunk_idx0 = b * 2 + 0;
        int32_t chunk_idx1 = b * 2 + 1;
        _mm_storel_epi64((__m128i *)(binary_buffer_0 + (size_t)chunk_idx0 * 8), pack_p_h0);
        _mm_storel_epi64((__m128i *)(binary_buffer_1 + (size_t)chunk_idx0 * 8), pack_n_h0);
        _mm_storel_epi64((__m128i *)(binary_buffer_0 + (size_t)chunk_idx1 * 8), pack_p_h1);
        _mm_storel_epi64((__m128i *)(binary_buffer_1 + (size_t)chunk_idx1 * 8), pack_n_h1);
#endif
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

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    const int nb = ne00 / QK_I2_S;
                    const int la_num = nb % 32;
                    const int groupla_num = nb % 32 != 0 ? 1 : 0;

                    auto quant_weight_ptr = src0_row + ir0 * nb01 / 4;

                    uint8_t array_buffer[6912]; // Keep original size for fallback

                    uint8_t binary_buffer_0[3456];
                    uint8_t binary_buffer_1[3456];

                    preprocess_weights(nb, quant_weight_ptr, array_buffer, binary_buffer_0, binary_buffer_1);

                    if (src0->type == GGML_TYPE_I2_S) {
                        float fused_val = 0.0f;
                        bitnet_vec_dot_fused(ne00,
                                             &fused_val,
                                             0,
                                             (uint8_t *)binary_buffer_0,
                                             0,
                                             (uint8_t *)binary_buffer_1,
                                             0,
                                             src1_col_de,
                                             0,
                                             1);

                        tmp[ir0 - iir0] = fused_val / (act_scales[i1]) * (*scale);
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

                memcpy(&dst_col[iir0], tmp, output_rows * sizeof(float));
            }
        }
    }
}
}
