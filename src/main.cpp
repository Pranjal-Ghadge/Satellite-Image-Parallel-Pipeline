#include <iostream>
#include <cstdlib>
#include <exception>
#include <limits>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "cpu_sequential.h"
#include "cpu_openmp.h"
#ifdef USE_CUDA
#include "gpu_cuda.h"
#endif

namespace {

void print_usage(const char* executable) {
    std::cout
        << "Usage:\n"
        << "  " << executable << " [image_path] [--threads N] [--resolution WIDTHxHEIGHT]\n"
        << "  " << executable << " --pre <path> --post <path> [--threads N]\n"
        << "  " << executable << " --help\n\n"
        << "Legacy single-image mode runs the Gaussian+Sobel benchmark.\n"
        << "Paired-image mode validates and converts both images to single-channel CV_32F; "
           "flood detection is not yet implemented.\n"
        << "Options:\n"
        << "  --pre <path>                 Pre-flood image (requires --post)\n"
        << "  --post <path>                Post-flood image (requires --pre)\n"
        << "  --threads N                  OpenMP thread count (legacy benchmark)\n"
        << "  --resolution WIDTHxHEIGHT    Resize legacy single-image input\n"
        << "  --help                       Show this help\n";
}

bool parse_positive_integer(const std::string& value, int& result) {
    try {
        size_t parsed = 0;
        const long parsed_value = std::stol(value, &parsed);
        if (parsed != value.size() || parsed_value <= 0 ||
            parsed_value > std::numeric_limits<int>::max()) {
            return false;
        }
        result = static_cast<int>(parsed_value);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

}

int main(int argc, char** argv) {
    std::string input_path = "data/raw/input_1024.jpg";
    std::string pre_path;
    std::string post_path;
    std::vector<int> threads_to_test = {2, 4, 8, 16};
    int target_width = 0;
    int target_height = 0;
    bool has_positional_input = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--pre" || arg == "--post" || arg == "--threads" || arg == "--resolution") {
            if (i + 1 >= argc) {
                std::cerr << "Error: Missing value for " << arg << ".\n";
                print_usage(argv[0]);
                return 2;
            }

            const std::string value = argv[++i];
            if (arg == "--pre") {
                pre_path = value;
            } else if (arg == "--post") {
                post_path = value;
            } else if (arg == "--threads") {
                int requested_threads = 0;
                if (!parse_positive_integer(value, requested_threads)) {
                    std::cerr << "Error: --threads must be a positive integer.\n";
                    return 2;
                }
                threads_to_test = {requested_threads};
            } else {
                const size_t separator = value.find('x');
                if (separator == std::string::npos ||
                    !parse_positive_integer(value.substr(0, separator), target_width) ||
                    !parse_positive_integer(value.substr(separator + 1), target_height)) {
                    std::cerr << "Error: --resolution must be WIDTHxHEIGHT with positive dimensions.\n";
                    return 2;
                }
            }
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "Error: Unknown option " << arg << ".\n";
            print_usage(argv[0]);
            return 2;
        } else {
            if (has_positional_input) {
                std::cerr << "Error: Only one positional image path is supported.\n";
                print_usage(argv[0]);
                return 2;
            }
            input_path = arg;
            has_positional_input = true;
        }
    }

    const bool has_pre_path = !pre_path.empty();
    const bool has_post_path = !post_path.empty();
    if (has_pre_path != has_post_path) {
        std::cerr << "Error: --pre and --post must be provided together.\n";
        print_usage(argv[0]);
        return 2;
    }
    if (has_pre_path && has_positional_input) {
        std::cerr << "Error: Use either a positional image path or --pre/--post, not both.\n";
        return 2;
    }

    if (has_pre_path) {
        cv::Mat pre_intensity;
        cv::Mat post_intensity;
        std::string error;
        if (!load_image_pair(pre_path, post_path, pre_intensity, post_intensity, error)) {
            std::cerr << "Error: " << error << std::endl;
            return 1;
        }

        std::cout << "Validated pre-/post-flood image pair: "
                  << pre_intensity.cols << "x" << pre_intensity.rows
                  << " pixels; both converted to single-channel CV_32F intensity.\n";
        return 0;
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