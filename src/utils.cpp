#include "utils.h"
#include <stdlib.h>
#include <iostream>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <cstdint>

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

std::vector<int8_t> unpack_i2_s(const uint8_t* data, size_t num_bytes) {
    std::vector<int8_t> unpacked_data;
    unpacked_data.reserve(num_bytes * 4);

    for (size_t i = 0; i < num_bytes; ++i) {
        uint8_t packed_byte = data[i];
        
        // Unpack the 4 x 2-bit values from the byte.
        // The packing order is high-bits to low-bits:
        // Bits 7,6 -> first value
        // Bits 5,4 -> second value
        // Bits 3,2 -> third value
        // Bits 1,0 -> fourth value
        // The values 0, 1, 2 correspond to a ternary system (-1, 0, 1),
        // while the value 3 is unused.
        unpacked_data.push_back(((packed_byte >> 6) & 0x03) - 1);
        unpacked_data.push_back(((packed_byte >> 4) & 0x03) - 1);
        unpacked_data.push_back(((packed_byte >> 2) & 0x03) - 1);
        unpacked_data.push_back(((packed_byte >> 0) & 0x03) - 1);
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
