#ifndef UTILS_H
#define UTILS_H

#include <cassert>
#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <array>

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
MatrixArrayPair<int, MAX_PERM_SIZE, MAX_SEG_SIZE> handle_block(std::vector<std::array<T, MAX_K>> mat_block, int k) {
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

VecPair<uint8_t> ternary_to_binary(std::vector<int8_t> ternary, size_t num_bytes);

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

    std::vector<std::array<int, MAX_PERM_SIZE>> permutations(output_rows);
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

        // Copy arrays directly
        permutations[i] = per_seg.a[0];
        segs[i] = per_seg.b[0];
    }

    return MatrixArrayPair<int, MAX_PERM_SIZE, MAX_SEG_SIZE>(permutations, segs);
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
rsr_forward(const std::vector<std::vector<int8_t>> &seg_sums, const std::vector<std::vector<int8_t>> bin_k, int k);

template <size_t MAX_SEGS, size_t MAX_K>
static std::array<float, MAX_K> RSRGemv(const std::array<int, MAX_SEGS> &vec, const std::vector<std::vector<int8_t>> &mat) {
    int mat_rows = mat.size();    // 256
    int mat_cols =mat[0].size(); // 8

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

template <size_t MAX_SEGS, size_t MAX_BLKS, size_t MAX_PERM, size_t MAX_SEG_SIZE, size_t MAX_K, size_t CHUNK_SIZE>
std::array<int, MAX_K * 2> rsr_inference(std::vector<int8_t> v,
                                         const std::vector<std::array<int, MAX_PERM>> &permutations,
                                         const std::vector<std::array<int, MAX_SEG_SIZE>> &segments,
                                         const std::vector<std::vector<int8_t>> bin_k,
                                         const int k) {
    int roundup = CHUNK_SIZE + (k - CHUNK_SIZE % k) % k;
    int n = v.size();

    assert((roundup / k) < MAX_BLKS); // warn to increase bound if needed

    thread_local std::array<std::array<int, MAX_SEGS>, MAX_BLKS> us = std::array<std::array<int, MAX_SEGS>, MAX_BLKS>();

    const int seg_size = 1 << k;
    std::array<int, MAX_PERM + 1> pref{};

    for (int i = 0; i < (roundup / k); i++) {
        const std::array<int, MAX_SEG_SIZE> &segment = segments[i];
        const std::array<int, MAX_PERM> &permutation = permutations[i];
        pref[0] = 0;

        // OPTIMIZE: < n && < seg_size
        for (int t = 0; t < n; ++t) {
            pref[t + 1] = pref[t] + static_cast<int>(v[permutation[t]]);
        }

        for (int j = 0; j < seg_size; j++) {
            int start = segment[j];
            int end;

            if (j + 1 < seg_size) {
                end = segment[j + 1];
            } else {
                end = n;
            }

            us[i][j] = pref[end] - pref[start];
        }
    }

    // Block product to Bin_k
    // TODO: change from here for RSR++
    std::array<int, MAX_K * 2> result;
    std::array<float, MAX_K> partial_result{}; // Changed to array

    for (size_t i = 0; i < std::min((size_t)roundup / k, permutations.size()) && i < MAX_BLKS; i++) {
        partial_result = RSRGemv<MAX_SEGS, MAX_K>(us[i], bin_k);

        for (int j = 0; j < k; j++) {
            result[i * k + j] = partial_result[j];
        }
    }

    return result;
}

std::vector<std::vector<int8_t>> seg_sum(std::vector<int8_t> v,
                                         const std::vector<std::vector<int8_t>> &perms,
                                         const std::vector<std::vector<int8_t>> &segs,
                                         int8_t k);

std::vector<std::vector<int8_t>> generateBinaryMatrix(int k);

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
