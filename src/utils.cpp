#include "utils.h"
#include <cstdio>
#include <cstring>

#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif
#include <stdlib.h>

#include <cassert>
#include <cmath>
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

vector<vector<int8_t>>
seg_sum(const vector<int8_t> &v, const vector<vector<int8_t>> &perms, const vector<vector<int8_t>> &segs, int8_t k) {
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
