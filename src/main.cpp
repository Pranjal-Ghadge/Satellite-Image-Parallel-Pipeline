#include <iostream>
#include <cstdlib>
#include <exception>
#include <limits>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "cpu_sequential.h"
#include "cpu_openmp.h"
#include "satellite_dataset.h"
#ifdef USE_CUDA
#include "gpu_cuda.h"
#endif

namespace {

int validate_pair_command(int argc, char** argv) {
    if (argc != 4 && argc != 6) {
        std::cerr << "Usage: parallel_pipeline --validate-pair <pre-flood.tif> "
                     "<post-flood.tif> [--band <one-based-index>]\n";
        return 2;
    }

    int band_index = 1;
    if (argc == 6) {
        if (std::string(argv[4]) != "--band") {
            std::cerr << "Expected --band before the band index.\n";
            return 2;
        }
        char* end = nullptr;
        const long parsed_band = std::strtol(argv[5], &end, 10);
        if (end == argv[5] || *end != '\0' || parsed_band < 1 ||
            parsed_band > std::numeric_limits<int>::max()) {
            std::cerr << "Band index must be a positive integer.\n";
            return 2;
        }
        band_index = static_cast<int>(parsed_band);
    }

    try {
        const SatelliteImagePair pair =
            load_aligned_sentinel1_pair(argv[2], argv[3], band_index);
        std::cout << "Pre-flood GeoTIFF: " << pair.pre_flood.source_path << '\n'
                  << "Post-flood GeoTIFF: " << pair.post_flood.source_path << '\n'
                  << "Aligned dimensions: " << pair.pre_flood.pixels.cols << " x "
                  << pair.pre_flood.pixels.rows << '\n'
                  << "Band: " << band_index << " ("
                  << pair.pre_flood.band_data_type << " / "
                  << pair.post_flood.band_data_type << " source samples)\n"
                  << "CRS and pixel geotransforms match; source geospatial metadata retained.\n";
    } catch (const std::exception& error) {
        std::cerr << "Satellite pair validation failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--validate-pair") {
        return validate_pair_command(argc, argv);
    }

    std::string input_path = "data/raw/input_1024.jpg";
    std::vector<int> threads_to_test = {2, 4, 8, 16};
    int target_width = 0;
    int target_height = 0;

    if (argc > 1) input_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--threads" && i + 1 < argc) {
            int requested_threads = std::atoi(argv[++i]);
            if (requested_threads > 0) {
                threads_to_test = {requested_threads};
            }
        } else if (arg == "--resolution" && i + 1 < argc) {
            std::string resolution = argv[++i];
            size_t separator = resolution.find('x');
            if (separator != std::string::npos) {
                target_width = std::atoi(resolution.substr(0, separator).c_str());
                target_height = std::atoi(resolution.substr(separator + 1).c_str());
            }
        }
    }

    cv::Mat input = cv::imread(input_path, cv::IMREAD_GRAYSCALE);

    if (input.empty()) {
        std::cerr << "Error: Could not load image from " << input_path << std::endl;
        std::cerr << "Please place a sample image in data/raw/ directory." << std::endl;
        return -1;
    }

    if (target_width > 0 && target_height > 0 &&
        (input.cols != target_width || input.rows != target_height)) {
        cv::resize(input, input, cv::Size(target_width, target_height), 0, 0, cv::INTER_AREA);
    }

    cv::Mat output_seq, output_omp;
#ifdef USE_CUDA
    cv::Mat output_cuda;
#endif
    Timer timer;

    std::cout << "\n==============================================" << std::endl;
    std::cout << " AAKASHNETRA : The Eye in the Sky" << std::endl;
    std::cout << " Parallel Satellite Image Pipeline Benchmark" << std::endl;
    std::cout << "==============================================" << std::endl;

    // 1. Sequential CPU Execution
    timer.start();
    process_sequential(input, output_seq);
    double seq_time = timer.stop_ms();
    log_performance("CPU Sequential Baseline", input.cols, input.rows, seq_time, seq_time, 1);
    cv::imwrite("data/output/seq_edge.jpg", output_seq);

    // 2. OpenMP Thread Scaling Benchmark
    for (int t : threads_to_test) {
        timer.start();
        process_openmp(input, output_omp, t);
        double omp_time = timer.stop_ms();
        log_performance("CPU OpenMP (" + std::to_string(t) + " Threads)", input.cols, input.rows, omp_time, seq_time, t);
    }
    cv::imwrite("data/output/omp_edge.jpg", output_omp);

#ifdef USE_CUDA
    // 3. CUDA GPU Execution
    timer.start();
    process_cuda(input, output_cuda);
    double cuda_time = timer.stop_ms();
    log_performance("GPU CUDA Acceleration", input.cols, input.rows, cuda_time, seq_time, 0);
    cv::imwrite("data/output/cuda_edge.jpg", output_cuda);
#endif

    std::cout << "\nExecution complete! Processed edge maps saved to 'data/output/' folder.\n" << std::endl;
    return 0;
}