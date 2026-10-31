#include "utils.h"

void log_performance(const std::string& mode, int width, int height, double execution_time_ms, double baseline_time_ms) {
    double total_pixels = static_cast<double>(width) * height;
    double time_sec = execution_time_ms / 1000.0;
    double megapixels_per_sec = (total_pixels / 1e6) / time_sec;

    std::cout << "\n------------------------------------------" << std::endl;
    std::cout << " Execution Mode: " << mode << std::endl;
    std::cout << " Resolution:     " << width << " x " << height << " px" << std::endl;
    std::cout << " Runtime:        " << execution_time_ms << " ms" << std::endl;
    std::cout << " Throughput:     " << megapixels_per_sec << " MP/s" << std::endl;

    if (baseline_time_ms > 0.0) {
        double speedup = baseline_time_ms / execution_time_ms;
        std::cout << " Speedup:        " << speedup << "x" << std::endl;
    }
    std::cout << "------------------------------------------" << std::endl;

    // Log data into CSV
    std::ofstream csv("docs/benchmark_results.csv", std::ios::app);
    csv << mode << "," << width << "x" << height << "," << execution_time_ms << "," << megapixels_per_sec << "\n";
    csv.close();
}