#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <utility>

using namespace std;

int binaryVectorToInt(const vector<int> &binaryVec);

template<typename T>
pair<std::vector<T>, std::vector<T>> handle_block(vector<vector<T>> mat_block);

void print_once(const std::string &message);

#endif // UTILS_H