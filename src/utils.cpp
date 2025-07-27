#include <iostream>
#include <mutex>
#include <string>

static void print_once(const std::string &message) {
    static std::once_flag logged_flag;
    std::call_once(logged_flag, [&message]() { std::cout << message << std::endl; });
}
