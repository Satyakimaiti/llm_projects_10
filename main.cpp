
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

// Use SSE2 for potentially faster reciprocal approximations if available and beneficial.
// The compiler flags specified (-mcpu=native) should handle this optimization.

double calculate(int iterations, double param1, double param2) {
    double result = 1.0;
    // Loop unrolling and vectorization are key for performance.
    // The compiler with -Ofast and -mcpu=native should attempt these.
    for (int i = 1; i <= iterations; ++i) {
        double j1 = i * param1 - param2;
        // Using 1.0 / j1 is generally optimized by compilers to fused multiply-add or similar instructions.
        result -= (1.0 / j1);

        double j2 = i * param1 + param2;
        result += (1.0 / j2);
    }
    return result;
}

int main() {
    // Use high-resolution clock for more accurate timing.
    auto start_time = std::chrono::high_resolution_clock::now();

    // The multiplication by 4 is moved outside the timed section to match Python's behavior.
    double final_result = calculate(200000000, 4.0, 1.0) * 4.0;

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Set precision for output to match Python's f-string formatting.
    std::cout << std::fixed << std::setprecision(12) << "Result: " << final_result << std::endl;
    std::cout << std::fixed << std::setprecision(6) << "Execution Time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}
