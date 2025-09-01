#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <cassert>
#include <cstdio>
#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif
#ifdef __x86_64__
#include <immintrin.h>
#endif
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../src/ska_sort.hpp"

void print_once(const std::string &message);

template <typename T> using matrix = std::vector<std::vector<T>>;

template <typename T> struct VecPair {
    std::vector<T> a;
    std::vector<T> b;

    VecPair(const std::vector<T> &a, const std::vector<T> &b) : a(a), b(b) {
    }
};

template <typename T, size_t MAX_SIZE_A, size_t MAX_SIZE_B> struct MatrixArrayPair {
    std::vector<std::array<T, MAX_SIZE_A>> a;
    std::vector<std::array<T, MAX_SIZE_B>> b;

    MatrixArrayPair() = default;
    MatrixArrayPair(const std::vector<std::array<T, MAX_SIZE_A>> &a, const std::vector<std::array<T, MAX_SIZE_B>> &b)
        : a(a), b(b) {
    }

    // Constructor for single arrays - wrap them in vectors
    MatrixArrayPair(const std::array<T, MAX_SIZE_A> &single_a, const std::array<T, MAX_SIZE_B> &single_b)
        : a({single_a}), b({single_b}) {
    }
};

template <typename T> struct MatrixPair {
    matrix<T> a;
    matrix<T> b;

    MatrixPair() = default;
    MatrixPair(const matrix<T> &a, const matrix<T> &b) : a(a), b(b) {
    }
};

class Timer {
  public:
    explicit Timer(const std::string &name = "")
        : name_(name), start_(std::chrono::high_resolution_clock::now()), stopped_(false) {
    }

    ~Timer() {
        if (!stopped_)
            log();
    }

    long long stop() {
        if (!stopped_) {
            auto end = std::chrono::high_resolution_clock::now();
            elapsed_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
            log();
            stopped_ = true;
        }
        return elapsed_ns_;
    }

    long long elapsed_ns() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(now - start_).count();
    }

  private:
    void log() const {
        std::ostringstream oss;
        oss << (name_.empty() ? "" : name_ + ": ") << elapsed_ns() << " ns\n";
        std::cerr << oss.str();
    }

    std::string name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    long long elapsed_ns_{0};
    bool stopped_;
};

struct PackedItem {
    uint16_t key;
    int index;

    PackedItem() = default;
    PackedItem(uint16_t k, int16_t i) : key(k), index(i) {
    }
};

template <typename T, size_t MAX_PERM_SIZE, size_t MAX_SEG_SIZE, size_t MAX_K>
MatrixArrayPair<int, MAX_PERM_SIZE, MAX_SEG_SIZE> handle_block(const std::vector<std::array<T, MAX_K>> &mat_block, int k) {
    assert(k < 16); // uint16_t being used

    int n = mat_block.size();

    if (n == 0) {
        return MatrixArrayPair<int, MAX_PERM_SIZE, MAX_SEG_SIZE>();
    }

    // Permutation
    std::array<int, MAX_PERM_SIZE> permutation;
    permutation.fill(-1);

    for (int i = 0; i < n; i++) {
        permutation[i] = i;
    }

    // Pre-pack binary vectors to integers for efficient sorting
    std::array<PackedItem, MAX_PERM_SIZE> packed_data;

    for (int16_t i = 0; i < n; i++) {
        uint16_t packed_value = 0;

        for (int j = 0; j < k; j++) {
            packed_value = (packed_value << 1) | static_cast<uint16_t>(mat_block[i][j]);
        }

        packed_data[i] = PackedItem(packed_value, i);
    }

    // Use ska_sort for radix sorting on packed integers
    ska_sort(packed_data.begin(), packed_data.begin() + n, [](const PackedItem &item) { return item.key; });

    // Extract permutation from sorted pairs
    for (int i = 0; i < n; i++) {
        permutation[i] = packed_data[i].index;
    }

    // Segmentation
    const int seg_size = 1 << k;
    std::array<int, MAX_SEG_SIZE> seg;

    seg.fill(-1);
    seg[0] = 0;

    for (int row = 0; row < n; row++) {
        auto value = packed_data[row].key; // Use already computed packed value
        if (seg[value] == -1) {
            seg[value] = row;
        }
    }

    if (seg[seg_size - 1] == -1) {
        seg[seg_size - 1] = n;
    }

    int last_one = seg[seg_size - 1];

    for (int i = seg_size - 2; i >= 0; i--) {
        if (seg[i] == -1) {
            seg[i] = last_one;
        }
        last_one = seg[i];
    }

    return MatrixArrayPair<int, MAX_PERM_SIZE, MAX_SEG_SIZE>(permutation, seg);
}

template <typename T, typename V>
std::vector<float> vectorMatrixMultiply(const std::vector<V> &vec, const std::vector<std::vector<T>> &mat) {
    int rows = mat.size();
    int cols = mat[0].size();
    std::vector<float> result(rows, 0);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i] += vec[j] * mat[i][j];
        }
    }

    return result;
}

// Optimized version that works with pointer to activations and matrix
template <typename T>
std::vector<float> vectorMatrixMultiply(const T* vec, const std::vector<std::vector<T>> &mat, int vec_size) {
    int rows = mat.size();
    int cols = std::min(vec_size, (int)mat[0].size());
    std::vector<float> result(rows, 0);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i] += vec[j] * mat[i][j];
        }
    }

    return result;
}

// Overload for vector<array> matrices
template <typename T, size_t N>
std::vector<float> vectorMatrixMultiply(const std::vector<T> &vec, const std::vector<std::array<T, N>> &mat) {
    int rows = mat.size();
    int cols = std::min((int)vec.size(), (int)N);
    std::vector<float> result(rows, 0);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i] += vec[j] * mat[i][j];
        }
    }

    return result;
}

/**
 * @brief Compute inverse permutation
 */
template <size_t MAX_PERM>
inline void compute_inverse_perm(std::array<int, MAX_PERM> &inv_perm, const std::array<int, MAX_PERM> &perm, int n) {
    for (int i = 0; i < n; i++) {
        inv_perm[perm[i]] = i;
    }
}

template <size_t MAX_K>
std::vector<std::array<int8_t, MAX_K>> generateBinaryMatrix(int k) {
    int rows = 1 << k;
    std::vector<std::array<int8_t, MAX_K>> matrix(rows); // Initialize matrix

    for (int i = 0; i < rows; ++i) {
        matrix[i].fill(0); // Initialize with zeros
        for (int j = 0; j < k; ++j) {
            // Generate the binary value for each position
            matrix[i][k - j - 1] = (i >> j) & 1; // Extract the j-th bit from i
        }
    }

    return matrix;
}

/**
 * @brief Unpacks a block of 2-bit quantized data into a vector of ternary values.
 *
 * Each byte in the input data contains four 2-bit values. This function extracts
 * these values and returns them as a vector of uint8_t, where each element is
 * a value from 0 to 2, representing a ternary system (-1, 0, 1).
 *
 * @param data A pointer to the packed 2-bit data.
 * @param num_bytes The number of bytes in the data block.
 * @return A vector of unpacked 8-bit integers, each representing a ternary value.
 */
std::vector<int8_t> unpack_i2_s(const uint8_t *data, size_t num_bytes);

VecPair<uint8_t> ternary_to_binary(const std::vector<int8_t> &ternary, size_t num_bytes);

template <size_t MAX_PERM_SIZE, size_t MAX_SEG_SIZE, size_t MAX_K, size_t MAX_BLOCKS>
MatrixArrayPair<int, MAX_PERM_SIZE, MAX_SEG_SIZE> preprocess(std::vector<std::vector<uint8_t>> &mat, int k) {
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

    std::vector<std::array<int, MAX_PERM_SIZE>> inv_permutations(output_rows);
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

        MatrixArrayPair<int, MAX_PERM_SIZE, MAX_SEG_SIZE> per_seg =
            handle_block<uint8_t, MAX_PERM_SIZE, MAX_SEG_SIZE, MAX_K>(block, k);

        // Compute inverse permutation directly and store it
        compute_inverse_perm<MAX_PERM_SIZE>(inv_permutations[i], per_seg.a[0], n);
        segs[i] = per_seg.b[0];
    }

    return MatrixArrayPair<int, MAX_PERM_SIZE, MAX_SEG_SIZE>(inv_permutations, segs);
}

/**
 * @brief Performs RSR forward pass: computes dot products and combines results
 *
 * @param seg_sums1 Segmented sums for first binary matrix
 * @param seg_sums2 Segmented sums for second binary matrix
 * @param bin_patterns1 Binary patterns for first matrix
 * @param bin_patterns2 Binary patterns for second matrix
 * @param k Block size parameter
 * @return Vector of float results (one per output row)
 */
std::vector<float>
rsr_forward(const std::vector<std::vector<int8_t>> &seg_sums, const std::vector<std::array<int8_t, 16>> &bin_k, int k);

template <size_t MAX_SEGS, size_t MAX_K>
static std::array<float, MAX_K> RSRGemv(const std::array<int, MAX_SEGS> &vec,
                                        const std::vector<std::array<int8_t, MAX_K>> &mat) {
    int mat_rows = mat.size(); // 256
    int mat_cols = MAX_K;      // Fixed size from array

    // Initialize result array
    std::array<float, MAX_K> result{};

    // gemv: (1x256) * (256x8) = (1x8)
    for (int i = 0; i < mat_cols && i < MAX_K; i++) {
        for (int j = 0; j < mat_rows && j < MAX_SEGS; j++) {
            result[i] += vec[j] * mat[j][i];
        }
    }

    return result;
}

/**
 * @brief Computes local prefix sum within a SIMD register using shift-and-add
 * ARM NEON version for 128-bit vectors (4 int32 elements)
 */
#ifdef __ARM_NEON__
inline int32x4_t prefix_sum_vec_neon(int32x4_t x) {
    // x = a, b, c, d
    // shift left by 1 element (4 bytes) and add
    x = vaddq_s32(x, vextq_s32(vdupq_n_s32(0), x, 3));
    // x = a, a+b, b+c, c+d
    // shift left by 2 elements (8 bytes) and add  
    x = vaddq_s32(x, vextq_s32(vdupq_n_s32(0), x, 2));
    // x = a, a+b, a+b+c, a+b+c+d
    return x;
}

/**
 * @brief Accumulates carry value to a block and returns new carry
 */
inline int32x4_t accumulate_neon(int32_t* p, int32x4_t carry) {
    // Broadcast the last element before adding carry
    int32x4_t last = vdupq_n_s32(p[3]);
    int32x4_t x = vld1q_s32(p);
    x = vaddq_s32(carry, x);
    vst1q_s32(p, x);
    return vaddq_s32(carry, last);
}
#endif

#ifdef __x86_64__
/**
 * @brief Computes local prefix sum within a SSE register
 */
inline __m128i prefix_sum_vec_sse(__m128i x) {
    x = _mm_add_epi32(x, _mm_slli_si128(x, 4));
    x = _mm_add_epi32(x, _mm_slli_si128(x, 8));
    return x;
}

/**
 * @brief Accumulates carry value to a block and returns new carry
 */
inline __m128i accumulate_sse(int32_t* p, __m128i carry) {
    __m128i last = _mm_set1_epi32(p[3]);
    __m128i x = _mm_loadu_si128((__m128i*)p);
    x = _mm_add_epi32(carry, x);
    _mm_storeu_si128((__m128i*)p, x);
    return _mm_add_epi32(carry, last);
}

#ifdef __AVX2__
/**
 * @brief Computes local prefix sum for 8 elements using AVX2
 * Note: AVX2 shifts work independently on two 128-bit lanes
 */
inline void prefix_sum_vec_avx2(int32_t* p) {
    __m256i x = _mm256_loadu_si256((__m256i*)p);
    x = _mm256_add_epi32(x, _mm256_slli_si256(x, 4));
    x = _mm256_add_epi32(x, _mm256_slli_si256(x, 8));
    _mm256_storeu_si256((__m256i*)p, x);
}
#endif
#endif

/**
 * @brief Optimized (dual) prefix sum using SIMD and inverse permutations
 * Scatters input directly to final positions for better cache locality
 */
template <size_t MAX_PERM>
inline void simd_dual_prefix_sum_inverse(std::array<int, MAX_PERM + 1> &output1,
                                         std::array<int, MAX_PERM + 1> &output2,
                                         const int8_t *input,
                                         const std::array<int, MAX_PERM> &inv_perm1,
                                         const std::array<int, MAX_PERM> &inv_perm2,
                                         int n) {
    output1[0] = 0;
    output2[0] = 0;

    // Scatter input directly using inverse permutations
    alignas(32) int32_t temp1[MAX_PERM];
    alignas(32) int32_t temp2[MAX_PERM];

    // Initialize to zero
    for (int i = 0; i < n; i++) {
        temp1[i] = 0;
        temp2[i] = 0;
    }

    // Scatter input values to their final positions
    for (int i = 0; i < n; i++) {
        temp1[inv_perm1[i]] = static_cast<int32_t>(input[i]);
    }
    for (int i = 0; i < n; i++) {
        temp2[inv_perm2[i]] = static_cast<int32_t>(input[i]);
    }

#ifdef __ARM_NEON__
    int i = 0;

    // Phase 1: Local prefix sums for both arrays
    for (; i + 3 < n; i += 4) {
        int32x4_t x1 = vld1q_s32(&temp1[i]);
        int32x4_t x2 = vld1q_s32(&temp2[i]);

        x1 = prefix_sum_vec_neon(x1);
        x2 = prefix_sum_vec_neon(x2);

        vst1q_s32(&temp1[i], x1);
        vst1q_s32(&temp2[i], x2);
    }

    // Handle remainder
    for (int j = i; j < n; j++) {
        if (j > 0) {
            temp1[j] += temp1[j - 1];
            temp2[j] += temp2[j - 1];
        }
    }

    // Phase 2: Accumulate across blocks for both arrays
    if (n > 4) {
        int32x4_t carry1 = vdupq_n_s32(temp1[3]);
        int32x4_t carry2 = vdupq_n_s32(temp2[3]);

        for (int j = 4; j + 3 < n; j += 4) {
            // Process both arrays to maximize cache reuse
            carry1 = accumulate_neon(&temp1[j], carry1);
            carry2 = accumulate_neon(&temp2[j], carry2);
        }

        // Handle final elements
        int last_carry1 = vgetq_lane_s32(carry1, 0);
        int last_carry2 = vgetq_lane_s32(carry2, 0);
        for (int j = (n / 4) * 4; j < n && j >= 4; j++) {
            temp1[j] += last_carry1;
            temp2[j] += last_carry2;
        }
    }

#elif defined(__AVX2__)
    int i = 0;

    // Phase 1: Local prefix sums
    for (; i + 7 < n; i += 8) {
        prefix_sum_vec_avx2(&temp1[i]);
        prefix_sum_vec_avx2(&temp2[i]);
    }

    // Handle remaining with SSE
    if (i + 3 < n) {
        __m128i x1 = _mm_loadu_si128((__m128i *)&temp1[i]);
        __m128i x2 = _mm_loadu_si128((__m128i *)&temp2[i]);
        x1 = prefix_sum_vec_sse(x1);
        x2 = prefix_sum_vec_sse(x2);
        _mm_storeu_si128((__m128i *)&temp1[i], x1);
        _mm_storeu_si128((__m128i *)&temp2[i], x2);
        i += 4;
    }

    // Final remainder
    for (int j = i; j < n; j++) {
        if (j > 0) {
            temp1[j] += temp1[j - 1];
            temp2[j] += temp2[j - 1];
        }
    }

    // Phase 2: Accumulate
    __m128i carry1 = _mm_setzero_si128();
    __m128i carry2 = _mm_setzero_si128();
    for (int j = 4; j < n; j += 4) {
        carry1 = accumulate_sse(&temp1[j], carry1);
        carry2 = accumulate_sse(&temp2[j], carry2);
    }

#elif defined(__SSE2__)
    int i = 0;

    // Phase 1: Local prefix sums
    for (; i + 3 < n; i += 4) {
        __m128i x1 = _mm_loadu_si128((__m128i *)&temp1[i]);
        __m128i x2 = _mm_loadu_si128((__m128i *)&temp2[i]);
        x1 = prefix_sum_vec_sse(x1);
        x2 = prefix_sum_vec_sse(x2);
        _mm_storeu_si128((__m128i *)&temp1[i], x1);
        _mm_storeu_si128((__m128i *)&temp2[i], x2);
    }

    // Remainder
    for (int j = i; j < n; j++) {
        if (j > 0) {
            temp1[j] += temp1[j - 1];
            temp2[j] += temp2[j - 1];
        }
    }

    // Phase 2: Accumulate
    if (n > 4) {
        __m128i carry1 = _mm_setzero_si128();
        __m128i carry2 = _mm_setzero_si128();
        for (int j = 4; j < n; j += 4) {
            carry1 = accumulate_sse(&temp1[j], carry1);
            carry2 = accumulate_sse(&temp2[j], carry2);
        }
    }

#else
    // Scalar fallback
    for (int i = 1; i < n; i++) {
        temp1[i] += temp1[i - 1];
        temp2[i] += temp2[i - 1];
    }
#endif

    // Copy results to output arrays
    for (int i = 0; i < n; i++) {
        output1[i + 1] = temp1[i];
        output2[i + 1] = temp2[i];
    }
}

/**
 * @brief Optimized (dual) prefix sum using SIMD via
 * [Algorithmica](https://en.algorithmica.org/hpc/algorithms/prefix/#vectorization) approach Processes permuted input
 * and writes to output array
 */
template <size_t MAX_PERM>
inline void simd_dual_prefix_sum(std::array<int, MAX_PERM + 1> &output1,
                                 std::array<int, MAX_PERM + 1> &output2,
                                 const int8_t *input,
                                 const std::array<int, MAX_PERM> &perm1,
                                 const std::array<int, MAX_PERM> &perm2,
                                 int n) {
    output1[0] = 0;
    output2[0] = 0;

    // Gather both permuted sequences
    alignas(32) int32_t temp1[MAX_PERM];
    alignas(32) int32_t temp2[MAX_PERM];

    for (int i = 0; i < n; i++) {
        temp1[i] = static_cast<int32_t>(input[perm1[i]]);
        temp2[i] = static_cast<int32_t>(input[perm2[i]]);
    }

#ifdef __ARM_NEON__
    int i = 0;

    // Phase 1: Local prefix sums for both arrays
    for (; i + 3 < n; i += 4) {
        int32x4_t x1 = vld1q_s32(&temp1[i]);
        int32x4_t x2 = vld1q_s32(&temp2[i]);

        x1 = prefix_sum_vec_neon(x1);
        x2 = prefix_sum_vec_neon(x2);

        vst1q_s32(&temp1[i], x1);
        vst1q_s32(&temp2[i], x2);
    }

    // Handle remainder
    for (int j = i; j < n; j++) {
        if (j > 0) {
            temp1[j] += temp1[j - 1];
            temp2[j] += temp2[j - 1];
        }
    }

    // Phase 2: Accumulate across blocks for both arrays
    if (n > 4) {
        int32x4_t carry1 = vdupq_n_s32(temp1[3]);
        int32x4_t carry2 = vdupq_n_s32(temp2[3]);

        for (int j = 4; j + 3 < n; j += 4) {
            // Process both arrays to maximize cache reuse
            carry1 = accumulate_neon(&temp1[j], carry1);
            carry2 = accumulate_neon(&temp2[j], carry2);
        }

        // Handle final elements
        int last_carry1 = vgetq_lane_s32(carry1, 0);
        int last_carry2 = vgetq_lane_s32(carry2, 0);
        for (int j = (n / 4) * 4; j < n && j >= 4; j++) {
            temp1[j] += last_carry1;
            temp2[j] += last_carry2;
        }
    }

#elif defined(__AVX2__)
    int i = 0;

    // Phase 1: Local prefix sums
    for (; i + 7 < n; i += 8) {
        prefix_sum_vec_avx2(&temp1[i]);
        prefix_sum_vec_avx2(&temp2[i]);
    }

    // Handle remaining with SSE
    if (i + 3 < n) {
        __m128i x1 = _mm_loadu_si128((__m128i *)&temp1[i]);
        __m128i x2 = _mm_loadu_si128((__m128i *)&temp2[i]);
        x1 = prefix_sum_vec_sse(x1);
        x2 = prefix_sum_vec_sse(x2);
        _mm_storeu_si128((__m128i *)&temp1[i], x1);
        _mm_storeu_si128((__m128i *)&temp2[i], x2);
        i += 4;
    }

    // Final remainder
    for (int j = i; j < n; j++) {
        if (j > 0) {
            temp1[j] += temp1[j - 1];
            temp2[j] += temp2[j - 1];
        }
    }

    // Phase 2: Accumulate
    __m128i carry1 = _mm_setzero_si128();
    __m128i carry2 = _mm_setzero_si128();
    for (int j = 4; j < n; j += 4) {
        carry1 = accumulate_sse(&temp1[j], carry1);
        carry2 = accumulate_sse(&temp2[j], carry2);
    }

#elif defined(__SSE2__)
    int i = 0;

    // Phase 1: Local prefix sums
    for (; i + 3 < n; i += 4) {
        __m128i x1 = _mm_loadu_si128((__m128i *)&temp1[i]);
        __m128i x2 = _mm_loadu_si128((__m128i *)&temp2[i]);
        x1 = prefix_sum_vec_sse(x1);
        x2 = prefix_sum_vec_sse(x2);
        _mm_storeu_si128((__m128i *)&temp1[i], x1);
        _mm_storeu_si128((__m128i *)&temp2[i], x2);
    }

    // Remainder
    for (int j = i; j < n; j++) {
        if (j > 0) {
            temp1[j] += temp1[j - 1];
            temp2[j] += temp2[j - 1];
        }
    }

    // Phase 2: Accumulate
    if (n > 4) {
        __m128i carry1 = _mm_setzero_si128();
        __m128i carry2 = _mm_setzero_si128();
        for (int j = 4; j < n; j += 4) {
            carry1 = accumulate_sse(&temp1[j], carry1);
            carry2 = accumulate_sse(&temp2[j], carry2);
        }
    }

#else
    // Scalar fallback
    cout << "Warning: Using non-vectorized (slower) prefix sum implementation.\n";
    for (int i = 1; i < n; i++) {
        temp1[i] += temp1[i - 1];
        temp2[i] += temp2[i - 1];
    }
#endif

    // Copy results to output arrays
    for (int i = 0; i < n; i++) {
        output1[i + 1] = temp1[i];
        output2[i + 1] = temp2[i];
    }
}

template <size_t MAX_SEGS, size_t MAX_BLKS, size_t MAX_PERM, size_t MAX_SEG_SIZE, size_t MAX_K, size_t CHUNK_SIZE>
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

    const int seg_size = 1 << k;
    std::array<int, CHUNK_SIZE * 2> result;

    // Precompute all prefix sums in 2D arrays using stored inverse permutations
    static thread_local std::array<std::array<int, MAX_PERM + 1>, MAX_BLKS> all_pref1;
    static thread_local std::array<std::array<int, MAX_PERM + 1>, MAX_BLKS> all_pref2;

    // Compute prefix sums using stored inverse permutations (no conversion needed)
    for (int i = 0; i < num_blocks; i++) {
        simd_dual_prefix_sum_inverse<MAX_PERM>(all_pref1[i], all_pref2[i], v, permutations1[i], permutations2[i], n);
    }

    for (int i = 0; i < num_blocks; i++) {
        std::array<float, MAX_K> partial_result{};
        const std::array<int, MAX_SEG_SIZE> &segment1 = segments1[i];
        const std::array<int, MAX_SEG_SIZE> &segment2 = segments2[i];
        const std::array<int, MAX_PERM + 1> &pref1 = all_pref1[i];
        const std::array<int, MAX_PERM + 1> &pref2 = all_pref2[i];

        for (int j = 0; j < seg_size; j++) {
            int start1 = segment1[j];
            int start2 = segment2[j];
            int end1;
            int end2;

            if (j + 1 < seg_size) {
                end1 = segment1[j + 1];
                end2 = segment2[j + 1];
            } else {
                end1 = n;
                end2 = n;
            }

            us[i][j] = (pref1[end1] - pref1[start1]) - (pref2[end2] - pref2[start2]);
        }

        partial_result = RSRGemv<MAX_SEGS, MAX_K>(us[i], bin_k);

        for (int j = 0; j < k; j++) {
            result[i * k + j] = partial_result[j];
        }
    }

    return result;
}

std::vector<std::vector<int8_t>> seg_sum(const std::vector<int8_t> &v,
                                         const std::vector<std::vector<int8_t>> &perms,
                                         const std::vector<std::vector<int8_t>> &segs,
                                         int8_t k);

/**
 * @brief Transposes a 2D matrix in-place.
 *
 * @tparam T The data type of matrix elements
 * @param matrix The matrix to transpose in-place (must be square)
 */
template <typename T> void matrix_transpose_inplace(std::vector<std::vector<T>> &matrix) {
    if (matrix.empty() || matrix[0].empty()) {
        return;
    }

    const size_t n = matrix.size();

    // Only works for square matrices
    assert(n == matrix[0].size() && "In-place transpose requires square matrix");

    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            std::swap(matrix[i][j], matrix[j][i]);
        }
    }
}

template <typename T> struct VectorHash {
    std::size_t operator()(const std::vector<T> &v) const {
        std::size_t seed = v.size();
        std::hash<T> hasher;
        for (const auto &x : v) {
            // FNV-1a / Boost-inspired hash combine
            seed ^= hasher(x) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

#endif
