#include "utils.h"
#include <array>
#include <cstdio>
#include <cstring>

#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif
#include <stdlib.h>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

void print_once(const std::string &message) {
    static once_flag logged_flag;
    call_once(logged_flag, [&message]() { cout << message << endl; });
}

vector<int8_t> unpack_i2_s(const uint8_t *data, size_t num_bytes) {
    const size_t BLOCK_SIZE = 32;
    const size_t num_full_blocks = num_bytes / BLOCK_SIZE;
    const size_t remaining_bytes = num_bytes % BLOCK_SIZE;

    if (remaining_bytes > 0) {
        throw std::runtime_error("Should not happen");
    }

    // Pre-allocate with exact size - no reallocation needed
    std::vector<int8_t> unpacked_data(num_bytes * 4);
    int8_t *output_ptr = unpacked_data.data();

    // Process full 32-byte blocks with interleaved layout
    for (size_t block = 0; block < num_full_blocks; ++block) {
        const uint8_t *block_data = data + block * BLOCK_SIZE;
        int8_t *block_output = output_ptr + block * 128; // 128 = BLOCK_SIZE * 4

        // Extract and write directly to output for maximum cache efficiency
        // Process 4 bytes at a time for better instruction-level parallelism
        for (size_t i = 0; i < BLOCK_SIZE; i += 4) {
            // Prefetch next cache line
            if (i + 8 < BLOCK_SIZE) {
                __builtin_prefetch(block_data + i + 8, 0, 3);
            }

            // Process 4 bytes in parallel
            uint32_t four_bytes = *reinterpret_cast<const uint32_t *>(block_data + i);

            // Extract all 16 values (4 bytes * 4 groups) in one loop
            for (size_t j = 0; j < 4; ++j) {
                uint8_t packed_byte = (four_bytes >> (j * 8)) & 0xFF;

                // BitNet interleaved layout within 128-weight blocks:
                // Bits 7,6 -> group 0 (weights 0-31)
                // Bits 5,4 -> group 1 (weights 32-63)
                // Bits 3,2 -> group 2 (weights 64-95)
                // Bits 1,0 -> group 3 (weights 96-127)
                block_output[i + j] = ((packed_byte >> 6) & 0x03);
                block_output[i + j + 32] = ((packed_byte >> 4) & 0x03);
                block_output[i + j + 64] = ((packed_byte >> 2) & 0x03);
                block_output[i + j + 96] = ((packed_byte >> 0) & 0x03);
            }
        }
    }

    return unpacked_data;
}

VecPair<uint8_t> ternary_to_binary(const vector<int8_t> &ternary, size_t num_bytes) {
    auto bin1 = vector<uint8_t>(num_bytes, 0);
    auto bin2 = vector<uint8_t>(num_bytes, 0);

    for (size_t i = 0; i < num_bytes; i++) {
        int8_t elem = (int8_t)ternary[i] - 1; // {0, 1, 2} ==> {-1, 0, 1}
        bin1[i] = !(elem == -1 || elem == 0);
        bin2[i] = !(elem == 0 || elem == 1);
    }

    return VecPair<uint8_t>(bin1, bin2);
}

inline void seg_sum(array<int, MAX_SEGS> &us,
                    const std::array<int, MAX_PERM + 1> prefix_sum_1,
                    const std::array<int, MAX_PERM + 1> prefix_sum_2,
                    const std::array<int, MAX_SEG_SIZE> &seg_1,
                    const std::array<int, MAX_SEG_SIZE> &seg_2,
                    const int k,
                    const int n) {
    const int seg_size = 1 << k; // 2 ** k <==> pow(2, MAX_K)

#ifdef __AVX2__
    // Process 8 seg_s at a time with AVX2
    int j = 0;
    for (; j + 7 < seg_size; j += 8) {
        // Load 8 seg_ start indices
        __m256i starts1 = _mm256_loadu_si256((__m256i *)&seg_1[j]);
        __m256i starts2 = _mm256_loadu_si256((__m256i *)&seg_2[j]);

        // Load 8 seg_ end indices
        __m256i ends1, ends2;
        if (j + 8 < seg_size) {
            ends1 = _mm256_loadu_si256((__m256i *)&seg_1[j + 1]);
            ends2 = _mm256_loadu_si256((__m256i *)&seg_2[j + 1]);
        } else {
            // Handle last seg_ boundary
            alignas(32) int temp_ends1[8], temp_ends2[8];
            for (int k = 0; k < 7; k++) {
                temp_ends1[k] = seg_1[j + k + 1];
                temp_ends2[k] = seg_2[j + k + 1];
            }
            temp_ends1[7] = n;
            temp_ends2[7] = n;
            ends1 = _mm256_load_si256((__m256i *)temp_ends1);
            ends2 = _mm256_load_si256((__m256i *)temp_ends2);
        }

        // Gather prefix_sum_ix sum values at seg_ boundaries
        __m256i prefix_sum_1_starts = _mm256_i32gather_epi32(&prefix_sum_1[0], starts1, 4);
        __m256i prefix_sum_1_ends = _mm256_i32gather_epi32(&prefix_sum_1[0], ends1, 4);
        __m256i prefix_sum_2_starts = _mm256_i32gather_epi32(&prefix_sum_2[0], starts2, 4);
        __m256i prefix_sum_2_ends = _mm256_i32gather_epi32(&prefix_sum_2[0], ends2, 4);

        // Compute seg_ sums: (prefix_sum_1[end] - prefix_sum_1[start]) - (prefix_sum_2[end] - prefix_sum_2[start])
        __m256i sum1 = _mm256_sub_epi32(prefix_sum_1_ends, prefix_sum_1_starts);
        __m256i sum2 = _mm256_sub_epi32(prefix_sum_2_ends, prefix_sum_2_starts);
        __m256i result = _mm256_sub_epi32(sum1, sum2);

        // Store 8 results
        _mm256_storeu_si256((__m256i *)&us[j], result);
    }

// Handle remaining seg_s with SSE
#ifdef __SSE4_1__
    for (; j + 3 < seg_size; j += 4) {
        __m128i starts1 = _mm_loadu_si128((__m128i *)&seg_1[j]);
        __m128i starts2 = _mm_loadu_si128((__m128i *)&seg_2[j]);

        __m128i ends1, ends2;
        if (j + 4 < seg_size) {
            ends1 = _mm_loadu_si128((__m128i *)&seg_1[j + 1]);
            ends2 = _mm_loadu_si128((__m128i *)&seg_2[j + 1]);
        } else {
            alignas(16) int temp_ends1[4], temp_ends2[4];
            for (int k = 0; k < 3 && j + k + 1 < seg_size; k++) {
                temp_ends1[k] = seg_1[j + k + 1];
                temp_ends2[k] = seg_2[j + k + 1];
            }
            temp_ends1[3] = n;
            temp_ends2[3] = n;
            ends1 = _mm_load_si128((__m128i *)temp_ends1);
            ends2 = _mm_load_si128((__m128i *)temp_ends2);
        }

        // SSE doesn't have gather, so extract and load manually
        alignas(16) int32_t gathered1_s[4], gathered1_e[4], gathered2_s[4], gathered2_e[4];
        for (int k = 0; k < 4; k++) {
            gathered1_s[k] = prefix_sum_1[((int *)&starts1)[k]];
            gathered1_e[k] = prefix_sum_1[((int *)&ends1)[k]];
            gathered2_s[k] = prefix_sum_2[((int *)&starts2)[k]];
            gathered2_e[k] = prefix_sum_2[((int *)&ends2)[k]];
        }

        __m128i prefix_sum_1_starts = _mm_load_si128((__m128i *)gathered1_s);
        __m128i prefix_sum_1_ends = _mm_load_si128((__m128i *)gathered1_e);
        __m128i prefix_sum_2_starts = _mm_load_si128((__m128i *)gathered2_s);
        __m128i prefix_sum_2_ends = _mm_load_si128((__m128i *)gathered2_e);

        __m128i sum1 = _mm_sub_epi32(prefix_sum_1_ends, prefix_sum_1_starts);
        __m128i sum2 = _mm_sub_epi32(prefix_sum_2_ends, prefix_sum_2_starts);
        __m128i result = _mm_sub_epi32(sum1, sum2);

        _mm_storeu_si128((__m128i *)&us[j], result);
    }
#endif

    // Scalar remainder
    for (; j < seg_size; j++) {
        int start1 = seg_1[j];
        int start2 = seg_2[j];
        int end1 = (j + 1 < seg_size) ? seg_1[j + 1] : n;
        int end2 = (j + 1 < seg_size) ? seg_2[j + 1] : n;
        us[j] = (prefix_sum_1[end1] - prefix_sum_1[start1]) - (prefix_sum_2[end2] - prefix_sum_2[start2]);
    }
#else
    for (int j = 0; j < seg_size; j++) {
        int start1 = seg_1[j];
        int start2 = seg_2[j];
        int end1;
        int end2;

        if (j + 1 < seg_size) {
            end1 = seg_1[j + 1];
            end2 = seg_2[j + 1];
        } else {
            end1 = n;
            end2 = n;
        }

        us[j] = (prefix_sum_1[end1] - prefix_sum_1[start1]) - (prefix_sum_2[end2] - prefix_sum_2[start2]);
    }
#endif
}

vector<float> rsr_forward(const vector<vector<int8_t>> &seg_sums, const vector<array<int8_t, 16>> &bin_k, int k) {
    vector<float> result = vector<float>(seg_sums.size() * k, 0.f);

    for (size_t i = 0; i < seg_sums.size(); i++) {
        vector<float> partial_results = vectorMatrixMultiply(seg_sums[i], bin_k);

        for (int j = 0; j < k; j++) {
            result[i * k + j] = partial_results[j];
        }
    }

    return result;
}

MatrixArrayPair<int, MAX_PERM, MAX_SEG_SIZE> preprocess(std::vector<std::vector<uint8_t>> &mat, int k) {
    int n = mat.size();
    int m = mat[0].size();

    int padding = (k - m % k) % k;

    if (padding != 0) {
        for (auto &row : mat) {
            row.resize(row.size() + padding, 0);
        }

        m += padding;
    }

    int output_rows = m / k;

    std::vector<std::array<int, MAX_PERM>> permutations(output_rows);
    std::vector<std::array<int, MAX_SEG_SIZE>> segs(output_rows);

    // Splitting into blocks (columnwise) for `handle_block`
    int start;
    int end;

    std::vector<std::array<uint8_t, MAX_K>> block(n);

    for (int i = 0; i < output_rows; i++) {
        start = i * k;
        end = start + k;

        for (int col = start; col < end; col++) {
            for (int row = 0; row < n; row++) {
                block[row][col - start] = mat[row][col];
            }
        }

        MatrixArrayPair<int, MAX_PERM, MAX_SEG_SIZE> per_seg = handle_block<uint8_t>(block, k);

        // Store normal permutation directly
        permutations[i] = per_seg.a[0];
        segs[i] = per_seg.b[0];
    }

    return MatrixArrayPair<int, MAX_PERM, MAX_SEG_SIZE>(permutations, segs);
}

std::array<int, CHUNK_SIZE * 2> rsr_inference_fused(const int8_t *v,
                                                    int v_size,
                                                    const std::vector<std::array<int, MAX_PERM>> &permutations1,
                                                    const std::vector<std::array<int, MAX_SEG_SIZE>> &segments1,
                                                    const std::vector<std::array<int, MAX_PERM>> &permutations2,
                                                    const std::vector<std::array<int, MAX_SEG_SIZE>> &segments2,
                                                    const std::vector<std::array<int8_t, MAX_K>> &bin_k,
                                                    const int k,
                                                    const int output_rows) {
    int roundup = output_rows + (k - output_rows % k) % k;
    int n = v_size;
    int num_blocks = roundup / k;

    assert(v_size <= (int)MAX_PERM);
    assert(num_blocks < MAX_BLKS); // warn to increase bound if needed

    static thread_local std::array<std::array<int, MAX_SEGS>, MAX_BLKS> us =
        std::array<std::array<int, MAX_SEGS>, MAX_BLKS>();

    std::array<int, CHUNK_SIZE * 2> result;

    // Precompute all prefix sums in 2D arrays using stored normal permutations
    static thread_local std::array<std::array<int, MAX_PERM + 1>, MAX_BLKS> all_pref1;
    static thread_local std::array<std::array<int, MAX_PERM + 1>, MAX_BLKS> all_pref2;

    for (int i = 0; i < num_blocks; i++) {
        simd_dual_prefix_sum_gather<MAX_PERM>(all_pref1[i], all_pref2[i], v, permutations1[i], permutations2[i], n);
    }

    for (int i = 0; i < num_blocks; i++) {
        const std::array<int, MAX_SEG_SIZE> &segment1 = segments1[i];
        const std::array<int, MAX_SEG_SIZE> &segment2 = segments2[i];
        const std::array<int, MAX_PERM + 1> &pref1 = all_pref1[i];
        const std::array<int, MAX_PERM + 1> &pref2 = all_pref2[i];

        seg_sum(us[i], pref1, pref2, segment1, segment2, k, n);
    }

    std::array<float, MAX_K> partial_result{};

    for (int i = 0; i < num_blocks; i++) {
        partial_result = RSRGemv<MAX_SEGS, MAX_K>(us[i], bin_k);

        for (int j = 0; j < k; j++) {
            result[i * k + j] = partial_result[j];
        }
    }

    return result;
}
