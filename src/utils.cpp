#include "utils.h"

#include <stdlib.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <mutex>
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

std::vector<int8_t> unpack_i2_s(const uint8_t *data, size_t num_bytes) {
    std::vector<int8_t> unpacked_data; // TODO: std::array optimization

    // Each 32-byte block contains 128 weights (4 weights per byte)
    // BitNet uses interleaved layout: 4 groups of 32 weights each
    const size_t block_size = num_bytes;             // 32 bytes per block
    const size_t weights_per_block = block_size * 4; // 4 * 32 = 128 weights
    const size_t num_blocks = num_bytes / block_size;

    unpacked_data.reserve(num_blocks * weights_per_block);

    for (size_t block = 0; block < num_blocks; ++block) {
        const uint8_t *block_data = data + block * block_size;

        // Temporary storage for the 4 interleaved groups
        std::vector<int8_t> group0(32), group1(32), group2(32), group3(32);

        // Extract the 4 interleaved groups from the 32-byte block
        for (size_t i = 0; i < block_size; ++i) {
            uint8_t packed_byte = block_data[i];

            // BitNet interleaved layout:
            // Bits 7,6 -> group 0 (weights 0-31)
            // Bits 5,4 -> group 1 (weights 32-63)
            // Bits 3,2 -> group 2 (weights 64-95)
            // Bits 1,0 -> group 3 (weights 96-127)
            // Values 0,1,2 map to -1,0,1 (ternary)
            group0[i] = ((packed_byte >> 6) & 0x03) - 1;
            group1[i] = ((packed_byte >> 4) & 0x03) - 1;
            group2[i] = ((packed_byte >> 2) & 0x03) - 1;
            group3[i] = ((packed_byte >> 0) & 0x03) - 1;
        }

        // Append groups in linear order to create sequential layout
        unpacked_data.insert(unpacked_data.end(), group0.begin(), group0.end());
        unpacked_data.insert(unpacked_data.end(), group1.begin(), group1.end());
        unpacked_data.insert(unpacked_data.end(), group2.begin(), group2.end());
        unpacked_data.insert(unpacked_data.end(), group3.begin(), group3.end());
    }

    return unpacked_data;
}

VecPair<uint8_t> ternary_to_binary(vector<int8_t> ternary, size_t num_bytes) {
    auto bin1 = vector<uint8_t>(num_bytes, 0);
    auto bin2 = vector<uint8_t>(num_bytes, 0);

    for (uint8_t i = 0; i < num_bytes; i++) {
        int8_t elem = (int8_t)ternary[i];
        bin1[i] = !(elem == -1 || elem == 0);
        bin2[i] = !(elem == 0 || elem == 1);
    }

    return VecPair<uint8_t>(bin1, bin2);
}

VecPair<vector<uint8_t>> preprocess(vector<vector<uint8_t>> &mat, int k) {
    int n = mat.size();

    // Padding
    int padding = (k - n % k) % k;
    for (auto &row : mat) {
        row.resize(row.size() + padding, 0);
    }

    for (int i = 0; i < padding; i++) {
        mat.push_back(vector<uint8_t>(n + padding, 0));
    }
    n = n + padding;

    vector<vector<uint8_t>> permutations(n / k, vector<uint8_t>(n));
    vector<vector<uint8_t>> segs(n / k, vector<uint8_t>(pow(2, k)));

    // Splitting into blocks (columnwise) for `handle_block`
    int start;
    int end;

    vector<vector<uint8_t>> block(n, vector<uint8_t>(k));

    for (int i = 0; i < n / k; i++) {
        // cout << "block " << i + 1 << " out of " << n / k << " blocks" << endl;
        start = i * k;
        end = start + k;
        for (int col = start; col < end; col++) {
            for (int row = 0; row < n; row++) {
                block[row][col - start] = mat[row][col];
            }
        }
        auto per_seg = handle_block(block);
        permutations[i] = per_seg.a;
        segs[i] = per_seg.b;
    }

    return VecPair<vector<uint8_t>>(permutations, segs);
}

vector<int> rsr_forward(vector<int> v,
                        const vector<vector<int>> &perms,
                        const vector<vector<int>> &segs,
                        vector<vector<int>> bin_k,
                        int k) {
    int n = perms[0].size();

    // Segmented Sums
    vector<vector<int>> us(perms.size(), vector<int>(pow(2, k)));

    int start, end;
    vector<int> segment, permutation;

    for (size_t i = 0; i < perms.size(); i++) {
        segment = segs[i];
        permutation = perms[i];

        for (size_t j = 0; j < segs.size(); j++) {
            start = segment[j];

            if (j < segment.size() - 1) {
                end = segment[j + 1];
            } else {
                end = n;
            }

            // Compute the segmented sum
            for (int index = start; index < end; index++) {
                us[i][j] += v[permutation[index]];
            }
        }
    }

    vector<int> result(n), partial_result;

    // Block product to Bin_k
    for (size_t i = 0; i < us.size(); i++) {
        partial_result = vectorMatrixMultiply(us[i], bin_k);

        for (int j = 0; j < k; j++) {
            result[i * k + j] = partial_result[j];
        }
    }

    return result;
}
