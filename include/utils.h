#ifndef UTILS_H
#define UTILS_H

#include <arm_neon.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

int binaryVectorToInt(const std::vector<int> &binaryVec);

void print_once(const std::string &message);

template <typename T> struct VecPair {
    std::vector<T> a;
    std::vector<T> b;

    VecPair(const std::vector<T> &a, const std::vector<T> &b) : a(a), b(b) {
    }
};

template <typename T> VecPair<int> handle_block(std::vector<std::vector<T>> mat_block) {
    int n = mat_block.size();
    if (n == 0) {
        return {{}, {}};
    }
    int k = mat_block[0].size();

    // Permutation
    std::vector<int> permutation(n);
    for (int i = 0; i < n; i++) {
        permutation[i] = i;
    }

    sort(permutation.begin(), permutation.end(), [&](int i, int j) {
        std::vector<int> vec_i(mat_block[i].begin(), mat_block[i].end());
        std::vector<int> vec_j(mat_block[j].begin(), mat_block[j].end());
        return binaryVectorToInt(vec_i) < binaryVectorToInt(vec_j);
    });

    // Segmentation
    std::vector<int> seg(pow(2, k), -1);
    seg[0] = 0;
    for (int row = 0; row < n; row++) {
        std::vector<int> current_row(mat_block[permutation[row]].begin(), mat_block[permutation[row]].end());
        int value = binaryVectorToInt(current_row);
        if (seg[value] == -1) {
            seg[value] = row;
        }
    }

    if (seg.size() > 0 && seg[seg.size() - 1] == -1) {
        seg[seg.size() - 1] = n;
    }

    int last_one = seg.empty() ? n : seg.back();

    for (int i = seg.size() - 2; i >= 0; i--) {
        if (seg[i] == -1) {
            seg[i] = last_one;
        }
        last_one = seg[i];
    }

    return {std::vector<int>(permutation.begin(), permutation.end()), std::vector<int>(seg.begin(), seg.end())};
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

VecPair<std::vector<int>> preprocess(std::vector<std::vector<uint8_t>> &mat, int k);

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

std::vector<int> rsr_inference(std::vector<int8_t> v,
                               const std::vector<std::vector<int>> &permutations,
                               const std::vector<std::vector<int>> &segments,
                               const std::vector<std::vector<int8_t>> bin_k,
                               const int k,
                               const int output_rows);

std::vector<std::vector<int8_t>> seg_sum(std::vector<int8_t> v,
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

std::vector<std::vector<int8_t>> generateBinaryMatrix(int k);

#endif
