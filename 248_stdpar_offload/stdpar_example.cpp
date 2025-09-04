#include <algorithm>
#include <execution>
#include <vector>
#include <cstdio>

int main() {
    std::vector<float> data(1 << 20, 1.0f);
    std::for_each(std::execution::par, data.begin(), data.end(), [](float &v) { v = v * 1.001f; });
    std::printf("stdpar done: %f\n", data[0]);
    return 0;
}
