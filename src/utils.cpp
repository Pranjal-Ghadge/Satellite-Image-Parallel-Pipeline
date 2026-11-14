#include "utils.h"

void log_performance(const std::string& mode, int width, int height, double execution_time_ms, double baseline_time_ms, int num_threads) {
    double total_pixels = static_cast<double>(width) * height;
    double time_sec = execution_time_ms / 1000.0;
    double megapixels_per_sec = (total_pixels / 1e6) / time_sec;

    std::cout << "\n------------------------------------------" << std::endl;
    std::cout << " Execution Mode: " << mode << std::endl;
    std::cout << " Resolution:     " << width << " x " << height << " px" << std::endl;
    std::cout << " Runtime:        " << execution_time_ms << " ms" << std::endl;
    std::cout << " Throughput:     " << megapixels_per_sec << " MP/s" << std::endl;

    double speedup = 1.0;
    double efficiency_pct = 100.0;
    if (baseline_time_ms > 0.0) {
        speedup = baseline_time_ms / execution_time_ms;
        std::cout << " Speedup:        " << speedup << "x" << std::endl;
        if (num_threads > 1) {
            efficiency_pct = (speedup / static_cast<double>(num_threads)) * 100.0;
            std::cout << " Efficiency:     " << efficiency_pct << "%" << std::endl;
        }
    }
    std::cout << "------------------------------------------" << std::endl;

    // Log data into CSV: mode, resolution, execution_time_ms, throughput_mpps, speedup, efficiency_pct
    std::ofstream csv("docs/benchmark_results.csv", std::ios::app);
    csv << mode << "," << width << "x" << height << "," << execution_time_ms << "," << megapixels_per_sec 
        << "," << speedup << "," << efficiency_pct << "\n";
    csv.close();
}
