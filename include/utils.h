
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

struct RSRConstants {
    static constexpr size_t MAX_SEGS = 1024;     // MAX_SEG_SIZE
    static constexpr size_t MAX_BLKS = 32;       // MAX_BLOCKS
    static constexpr size_t MAX_PERM = 8192;     // MAX_PERM_SIZE
    static constexpr size_t MAX_SEG_SIZE = 1024; // MAX_SEG_SIZE
    static constexpr size_t MAX_K = 10;          // MAX_K
    static constexpr size_t CHUNK_SIZE = 128;    // CHUNK_SIZE
};

// Import RSRConstants values for direct use
constexpr auto MAX_PERM = RSRConstants::MAX_PERM;
constexpr auto MAX_SEG_SIZE = RSRConstants::MAX_SEG_SIZE;
constexpr auto MAX_K = RSRConstants::MAX_K;
constexpr auto CHUNK_SIZE = RSRConstants::CHUNK_SIZE;
constexpr auto MAX_SEGS = RSRConstants::MAX_SEGS;
constexpr auto MAX_BLKS = RSRConstants::MAX_BLKS;

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

template <typename T>
MatrixArrayPair<int, MAX_PERM, MAX_SEG_SIZE> handle_block(const std::vector<std::array<T, MAX_K>> &mat_block, int k) {
    assert(k < 16); // uint16_t being used

    int n = mat_block.size();

    if (n == 0) {
        return MatrixArrayPair<int, MAX_PERM, MAX_SEG_SIZE>();
    }

    // Permutation
    std::array<int, MAX_PERM> permutation;
    permutation.fill(-1);

    for (int i = 0; i < n; i++) {
        permutation[i] = i;
    }

    // Pre-pack binary vectors to integers for efficient sorting
    std::array<PackedItem, MAX_PERM> packed_data;

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

    return MatrixArrayPair<int, MAX_PERM, MAX_SEG_SIZE>(permutation, seg);
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
std::vector<float> vectorMatrixMultiply(const T *vec, const std::vector<std::vector<T>> &mat, int vec_size) {
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

template <size_t MAX_K> std::vector<std::array<int8_t, MAX_K>> generateBinaryMatrix(int k) {
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

MatrixArrayPair<int, MAX_PERM, MAX_SEG_SIZE> preprocess(std::vector<std::vector<uint8_t>> &mat, int k);

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

template <size_t MAX_PERM>
inline void simd_dual_prefix_sum_gather(std::array<int, MAX_PERM + 1> &output1,
                                        std::array<int, MAX_PERM + 1> &output2,
                                        const int8_t *input,
                                        const std::array<int, MAX_PERM> &perm1, // Normal perms
                                        const std::array<int, MAX_PERM> &perm2,
                                        int n) {
    output1[0] = 0;
    output2[0] = 0;

    alignas(32) int32_t temp1[MAX_PERM];
    alignas(32) int32_t temp2[MAX_PERM];

#ifdef __AVX2__
    int i = 0;
    for (; i + 7 < n; i += 8) {
        // Load 8 permutation indices
        __m256i indices1 = _mm256_loadu_si256((__m256i *)&perm1[i]);
        __m256i indices2 = _mm256_loadu_si256((__m256i *)&perm2[i]);

        // GATHER 8 int8 values at once using AVX2!
        __m256i gathered1 = _mm256_i32gather_epi32((const int *)input, indices1, 1);
        __m256i gathered2 = _mm256_i32gather_epi32((const int *)input, indices2, 1);

        // Extract int8 and sign-extend to int32
        gathered1 = _mm256_srai_epi32(_mm256_slli_epi32(gathered1, 24), 24);
        gathered2 = _mm256_srai_epi32(_mm256_slli_epi32(gathered2, 24), 24);

        // Store sequentially (no scatter needed!)
        _mm256_store_si256((__m256i *)&temp1[i], gathered1);
        _mm256_store_si256((__m256i *)&temp2[i], gathered2);
    }

    // Handle remainder
    for (; i < n; i++) {
        temp1[i] = static_cast<int32_t>(input[perm1[i]]);
        temp2[i] = static_cast<int32_t>(input[perm2[i]]);
    }

    // Phase 1: Vectorized local prefix sums using AVX2
    i = 0;
    for (; i + 7 < n; i += 8) {
        __m256i x1 = _mm256_load_si256((__m256i *)&temp1[i]);
        __m256i x2 = _mm256_load_si256((__m256i *)&temp2[i]);

        // AVX2 prefix sum within 256-bit register
        // Note: AVX2 shifts work on 128-bit lanes independently
        x1 = _mm256_add_epi32(x1, _mm256_slli_si256(x1, 4));
        x1 = _mm256_add_epi32(x1, _mm256_slli_si256(x1, 8));

        x2 = _mm256_add_epi32(x2, _mm256_slli_si256(x2, 4));
        x2 = _mm256_add_epi32(x2, _mm256_slli_si256(x2, 8));

        _mm256_store_si256((__m256i *)&temp1[i], x1);
        _mm256_store_si256((__m256i *)&temp2[i], x2);
    }

    // Handle remaining with SSE
    if (i + 3 < n) {
        __m128i x1 = _mm_load_si128((__m128i *)&temp1[i]);
        __m128i x2 = _mm_load_si128((__m128i *)&temp2[i]);

        x1 = _mm_add_epi32(x1, _mm_slli_si128(x1, 4));
        x1 = _mm_add_epi32(x1, _mm_slli_si128(x1, 8));

        x2 = _mm_add_epi32(x2, _mm_slli_si128(x2, 4));
        x2 = _mm_add_epi32(x2, _mm_slli_si128(x2, 8));

        _mm_store_si128((__m128i *)&temp1[i], x1);
        _mm_store_si128((__m128i *)&temp2[i], x2);
        i += 4;
    }

    // Scalar remainder
    for (int j = i; j < n; j++) {
        if (j > 0) {
            temp1[j] += temp1[j - 1];
            temp2[j] += temp2[j - 1];
        }
    }

    // Phase 2: Accumulate across blocks
    __m128i carry1 = _mm_setzero_si128();
    __m128i carry2 = _mm_setzero_si128();

    for (int j = 4; j < n; j += 4) {
        __m128i last1 = _mm_set1_epi32(temp1[j + 3]);
        __m128i last2 = _mm_set1_epi32(temp2[j + 3]);

        __m128i x1 = _mm_load_si128((__m128i *)&temp1[j]);
        __m128i x2 = _mm_load_si128((__m128i *)&temp2[j]);

        x1 = _mm_add_epi32(carry1, x1);
        x2 = _mm_add_epi32(carry2, x2);

        _mm_store_si128((__m128i *)&temp1[j], x1);
        _mm_store_si128((__m128i *)&temp2[j], x2);

        carry1 = _mm_add_epi32(carry1, last1);
        carry2 = _mm_add_epi32(carry2, last2);
    }

    // Vectorized copy to output using AVX2
    for (i = 0; i + 7 < n; i += 8) {
        __m256i v1 = _mm256_load_si256((__m256i *)&temp1[i]);
        __m256i v2 = _mm256_load_si256((__m256i *)&temp2[i]);
        _mm256_storeu_si256((__m256i *)&output1[i + 1], v1);
        _mm256_storeu_si256((__m256i *)&output2[i + 1], v2);
    }

    // Copy remaining
    for (; i < n; i++) {
        output1[i + 1] = temp1[i];
        output2[i + 1] = temp2[i];
    }

#else
    // Scalar fallback
    for (int i = 0; i < n; i++) {
        temp1[i] = static_cast<int32_t>(input[perm1[i]]);
        temp2[i] = static_cast<int32_t>(input[perm2[i]]);
    }

    for (int i = 1; i < n; i++) {
        temp1[i] += temp1[i - 1];
        temp2[i] += temp2[i - 1];
    }

    // Copy results to output arrays
    for (int i = 0; i < n; i++) {
        output1[i + 1] = temp1[i];
        output2[i + 1] = temp2[i];
    }
#endif
}

std::array<int, CHUNK_SIZE * 2> rsr_inference_fused(const int8_t *v,
                                                    int v_size,
                                                    const std::vector<std::array<int, MAX_PERM>> &permutations1,
                                                    const std::vector<std::array<int, MAX_SEG_SIZE>> &segments1,
                                                    const std::vector<std::array<int, MAX_PERM>> &permutations2,
                                                    const std::vector<std::array<int, MAX_SEG_SIZE>> &segments2,
                                                    const std::vector<std::array<int8_t, MAX_K>> &bin_k,
                                                    const int k,
                                                    const int output_rows);

void seg_sum(std::array<int, MAX_SEGS> &us,
             const std::array<int, MAX_PERM + 1> prefix_sum_1,
             const std::array<int, MAX_PERM + 1> prefix_sum_2,
             const std::array<int, MAX_SEG_SIZE> &seg_1,
             const std::array<int, MAX_SEG_SIZE> &seg_2,
             const int k,
             const int n);

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
