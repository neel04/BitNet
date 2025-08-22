#include "utils.h"

#include <arm_neon.h>
#include <stdlib.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

int binaryVectorToInt(const vector<int> &binaryVec) {
    int result = 0;
    int n = binaryVec.size();

    for (int i = 0; i < n; ++i) {
        result = (result << 1) | binaryVec[i]; // Left-shift result and add the next bit
    }

    return result;
}

void print_once(const std::string &message) {
    static once_flag logged_flag;
    call_once(logged_flag, [&message]() { cout << message << endl; });
}

vector<int8_t> unpack_i2_s(const uint8_t *data, size_t num_bytes) {
    std::vector<int8_t> unpacked_data;

    const size_t BLOCK_SIZE = 32;

    // Process full blocks
    const size_t num_full_blocks = num_bytes / BLOCK_SIZE;
    const size_t remaining_bytes = num_bytes % BLOCK_SIZE;

    unpacked_data.reserve(num_bytes * 4); // 4 weights per byte

    // Process full 32-byte blocks with interleaved layout
    for (size_t block = 0; block < num_full_blocks; ++block) {
        const uint8_t *block_data = data + block * BLOCK_SIZE;

        // Temporary storage for the 4 interleaved groups
        std::vector<int8_t> group0(32), group1(32), group2(32), group3(32);

        // Extract the 4 interleaved groups from the 32-byte block
        for (size_t i = 0; i < BLOCK_SIZE; ++i) {
            uint8_t packed_byte = block_data[i];

            // BitNet interleaved layout within 128-weight blocks:
            // Bits 7,6 -> group 0 (weights 0-31)
            // Bits 5,4 -> group 1 (weights 32-63)
            // Bits 3,2 -> group 2 (weights 64-95)
            // Bits 1,0 -> group 3 (weights 96-127)
            group0[i] = ((packed_byte >> 6) & 0x03);
            group1[i] = ((packed_byte >> 4) & 0x03);
            group2[i] = ((packed_byte >> 2) & 0x03);
            group3[i] = ((packed_byte >> 0) & 0x03);
        }

        // Append groups in linear order to create sequential layout
        unpacked_data.insert(unpacked_data.end(), group0.begin(), group0.end());
        unpacked_data.insert(unpacked_data.end(), group1.begin(), group1.end());
        unpacked_data.insert(unpacked_data.end(), group2.begin(), group2.end());
        unpacked_data.insert(unpacked_data.end(), group3.begin(), group3.end());
    }

    if (remaining_bytes > 0) {
        // TODO: So this shouldn't happen at all AIUI
        throw std::runtime_error("Should not happen");
    }

    return unpacked_data;
}

VecPair<uint8_t> ternary_to_binary(vector<int8_t> ternary, size_t num_bytes) {
    auto bin1 = vector<uint8_t>(num_bytes, 0);
    auto bin2 = vector<uint8_t>(num_bytes, 0);

    for (size_t i = 0; i < num_bytes; i++) {
        int8_t elem = (int8_t)ternary[i] - 1; // {0, 1, 2} ==> {-1, 0, 1}
        bin1[i] = !(elem == -1 || elem == 0);
        bin2[i] = !(elem == 0 || elem == 1);
    }

    return VecPair<uint8_t>(bin1, bin2);
}

VecPair<vector<int>> preprocess(vector<vector<uint8_t>> &mat, int k) {
    int n = mat.size();
    int m = mat[0].size();

    // Padding
    int padding = (k - m % k) % k;
    for (auto &row : mat) {
        row.resize(row.size() + padding, 0);
    }

    // for (int i = 0; i < padding; i++) {
    //     mat.push_back(vector<uint8_t>(m + padding, 0));
    // }
    m += padding;

    vector<vector<int>> permutations(n / k, vector<int>(n));
    vector<vector<int>> segs(n / k, vector<int>(pow(2, k)));

    // Splitting into blocks (columnwise) for `handle_block`
    int start;
    int end;

    vector<vector<uint8_t>> block(n, vector<uint8_t>(k));

    for (int i = 0; i < std::min(n, (int)mat[0].size()) / k; i++) {
        // cout << "block " << i + 1 << " out of " << n / k << " blocks" << endl;
        start = i * k;
        end = start + k;
        for (int col = start; col < end; col++) {
            for (int row = 0; row < n; row++) {
                block[row][col - start] = mat[row][col];
            }
        }
        VecPair<int> per_seg = handle_block(block);
        permutations[i].assign(per_seg.a.begin(), per_seg.a.end());
        segs[i].assign(per_seg.b.begin(), per_seg.b.end());
    }

    return VecPair<vector<int>>(permutations, segs);
}

vector<vector<int8_t>>
seg_sum(vector<int8_t> v, const vector<vector<int8_t>> &perms, const vector<vector<int8_t>> &segs, int8_t k) {
    int8_t n = perms[0].size();

    // Segmented Sums
    vector<vector<int8_t>> us(perms.size(), vector<int8_t>(pow(2, k)));

    uint8_t start, end;
    vector<int8_t> segment, permutation;

    for (size_t i = 0; i < perms.size(); i++) {
        segment = segs[i];
        permutation = perms[i];

        for (size_t j = 0; j < segment.size(); j++) {
            start = segment[j];

            if (j < segment.size() - 1) {
                end = segment[j + 1];
            } else {
                end = n;
            }

            // Compute the segmented sum
            for (size_t index = start; index < end; index++) {
                us[i][j] += v[permutation[index]];
            }
        }
    }

    return us;
}

vector<float> rsr_forward(const vector<vector<int8_t>> &seg_sums, const vector<vector<int8_t>> bin_k, int k) {
    vector<float> result = vector<float>(seg_sums.size() * k, 0.f);

    for (size_t i = 0; i < seg_sums.size(); i++) {
        vector<float> partial_results = vectorMatrixMultiply(seg_sums[i], bin_k);

        for (int j = 0; j < k; j++) {
            result[i * k + j] = partial_results[j];
        }
    }

    return result;
}

vector<vector<int8_t>> generateBinaryMatrix(int k) {
    int rows = pow(2, k);                                      // 2^k rows
    vector<vector<int8_t>> matrix(rows, vector<int8_t>(k, 0)); // Initialize matrix with 0s

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < k; ++j) {
            // Generate the binary value for each position
            matrix[i][k - j - 1] = (i >> j) & 1; // Extract the j-th bit from i
        }
    }

    return matrix;
}


static vector<float> RSRGemv(const vector<int>& vec, const vector<vector<int8_t>>& mat) {
    int vec_size = vec.size();        // 256
    int mat_rows = mat.size();        // 256 
    int mat_cols = mat[0].size();     // 8

    // Initialize result with matrix columns, not vector size
    vector<float> result(mat_cols, 0);

    // Perform vector-matrix multiplication: (1x256) * (256x8) = (1x8)
    for (int i = 0; i < mat_cols; i++) {
        for (int j = 0; j < mat_rows && j < vec_size; j++) {
            result[i] += vec[j] * mat[j][i];
        }
    }

    return result;
}

vector<int> rsr_inference(vector<int8_t> v,
                            const vector<vector<int>> &permutations,
                            const vector<vector<int>> &segments,
                            const vector<vector<int8_t>> bin_k,
                            const int k,
                            const int output_rows) {
    int n = permutations[0].size();

    // segmented sums
    vector<vector<int>> us(permutations.size(), vector<int>(pow(2, k)));

    int start;
    int end;
    vector<int> segment;
    vector<int> permutation;

    for (size_t i = 0; i < permutations.size(); i++) {
        segment = segments[i];
        permutation = permutations[i];

        // Each block
        for (size_t j = 0; j < segment.size(); j++) {
            start = segment[j];
            if (j < segment.size() - 1) {
                end = segment[j + 1];
            } else {
                end = n;
            }
            // Segmented sum
            for (int index = start; index < end; index++) {
                us[i][j] += v[permutation[index]];
            }           
        }
    }

    int roundup = output_rows + (k - output_rows % k) % k;

    // Block product to Bin_k
    // TODO: change from here for RSR++
    vector<int> result(roundup); // n
    vector<float> partial_result; // was: `int`

    for (size_t i = 0; i < std::min((size_t)roundup / k, us.size()); i++) {
        partial_result = RSRGemv(us[i], bin_k);
        for (int j = 0; j < k; j++) {
            result[i * k + j] = partial_result[j];
        }
    }
    return result;
}
