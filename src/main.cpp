#include <iostream>
#include <cstdlib>
#include <vector>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "cpu_sequential.h"
#include "cpu_openmp.h"
#ifdef USE_CUDA
#include "gpu_cuda.h"
#endif

int main(int argc, char** argv) {
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
    std::cout << " Starting Parallel Satellite Pipeline Benchmark" << std::endl;
    std::cout << "==============================================" << std::endl;

    // 1. Sequential CPU Execution
    timer.start();
    process_sequential(input, output_seq);
    double seq_time = timer.stop_ms();
    log_performance("CPU Sequential Baseline", input.cols, input.rows, seq_time);
    cv::imwrite("data/output/seq_edge.jpg", output_seq);

    // 2. OpenMP Thread Scaling Benchmark
    for (int t : threads_to_test) {
        timer.start();
        process_openmp(input, output_omp, t);
        double omp_time = timer.stop_ms();
        log_performance("CPU OpenMP (" + std::to_string(t) + " Threads)", input.cols, input.rows, omp_time, seq_time);
    }
    cv::imwrite("data/output/omp_edge.jpg", output_omp);

#ifdef USE_CUDA
    // 3. CUDA GPU Execution
    timer.start();
    process_cuda(input, output_cuda);
    double cuda_time = timer.stop_ms();
    log_performance("GPU CUDA Acceleration", input.cols, input.rows, cuda_time, seq_time);
    cv::imwrite("data/output/cuda_edge.jpg", output_cuda);
#endif

    std::cout << "\nExecution complete! Processed edge maps saved to 'data/output/' folder.\n" << std::endl;
    return 0;
}