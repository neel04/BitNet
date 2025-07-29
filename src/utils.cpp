#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <mutex>
#include <string>
#include <utility>
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

template<typename T>
pair<std::vector<T>, std::vector<T>> handle_block(vector<vector<T>> mat_block) {
    int n = mat_block.size();
    int k = mat_block[0].size();

    // Permutation
    vector<int> permutation(n);
    for (int i = 0; i < n; i++) {
        permutation[i] = i;
    }

    sort(permutation.begin(), permutation.end(), [&](int i, int j) {
        // return vec[i] < vec[j]
        return binaryVectorToInt(mat_block[i]) < binaryVectorToInt(mat_block[j]);
    });

    // Segmentation
    vector<int> seg(pow(2, k), -1);
    seg[0] = 0;
    for (int row = 0; row < n; row++) {
        int value = binaryVectorToInt(mat_block[permutation[row]]);
        if (seg[value] == -1) {
            seg[value] = row;
        }
    }
    if (seg[seg.size() - 1] == -1) {
        seg[seg.size() - 1] = n;
    }
    int last_one = seg[seg.size() - 1];
    for (int i = seg.size() - 2; i >= 0; i--) {
        if (seg[i] == -1) {
            seg[i] = last_one;
        }
        last_one = seg[i];
    }

    return make_pair(permutation, seg);
}

void print_once(const std::string &message) {
    static once_flag logged_flag;
    call_once(logged_flag, [&message]() { cout << message << endl; });
}
