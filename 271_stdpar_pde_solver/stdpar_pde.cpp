#include <algorithm>
#include <execution>
#include <vector>
#include <cstdio>

int main() {
    int nx = 256;
    int ny = 256;
    std::vector<float> grid(nx * ny, 0.0f);
    std::vector<float> next(nx * ny, 0.0f);

    std::for_each(std::execution::par, grid.begin(), grid.end(), [](float &v) { v = 1.0f; });

    for (int step = 0; step < 10; ++step) {
        std::for_each(std::execution::par, grid.begin(), grid.end(), [&](float &v) {
            v = v * 0.99f + 0.01f;
        });
    }

    std::printf("stdpar PDE done %f\n", grid[0]);
    return 0;
}
