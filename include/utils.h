#ifndef UTILS_H
#define UTILS_H

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

template <typename T> VecPair<T> handle_block(std::vector<std::vector<T>> mat_block) {
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

    return {std::vector<T>(permutation.begin(), permutation.end()), std::vector<T>(seg.begin(), seg.end())};
}

template <typename T>
std::vector<int8_t> vectorMatrixMultiply(const std::vector<T> &vec, const std::vector<std::vector<T>> &mat) {
    int n = vec.size();

    // Initialize the result vector with zeros
    std::vector<int8_t> result(n, 0);

    // Perform vector-matrix multiplication
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i] += vec[j] * mat[j][i];
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

VecPair<std::vector<int8_t>> preprocess(std::vector<std::vector<uint8_t>> &mat, int k);

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
template <typename T> void matrix_transpose_inplace(std::vector<std::vector<T>> &matrix);

std::vector<std::vector<int8_t>> generateBinaryMatrix(int k);

#endif
