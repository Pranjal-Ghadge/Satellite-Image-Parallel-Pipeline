#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>

// Spatial Kernels
const int GAUSS_KERNEL[5][5] = {
    {1,  4,  7,  4, 1},
    {4, 16, 26, 16, 4},
    {7, 26, 41, 26, 7},
    {4, 16, 26, 16, 4},
    {1,  4,  7,  4, 1}
};
const int GAUSS_SUM = 273;

const int SOBEL_GX[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

const int SOBEL_GY[3][3] = {
    {-1, -2, -1},
     {0,  0,  0},
     {1,  2,  1}
};

// Stage 1: 5x5 Gaussian Filter Function
void apply_gaussian_blur(const cv::Mat& input, cv::Mat& output) {
    int rows = input.rows;
    int cols = input.cols;

    // Initialize border pixels to zero
    output = cv::Mat::zeros(rows, cols, CV_8UC1);

    // Row pointer iteration for cache locality
    for (int r = 2; r < rows - 2; ++r) {
        uchar* out_ptr = output.ptr<uchar>(r);
        for (int c = 2; c < cols - 2; ++c) {
            int sum = 0;
            for (int kr = -2; kr <= 2; ++kr) {
                const uchar* in_row = input.ptr<uchar>(r + kr);
                for (int kc = -2; kc <= 2; ++kc) {
                    sum += in_row[c + kc] * GAUSS_KERNEL[kr + 2][kc + 2];
                }
            }
            out_ptr[c] = static_cast<uchar>(sum / GAUSS_SUM);
        }
    }
}

// Stage 2: 3x3 Sobel Edge Detection Function
void apply_sobel_edge_detection(const cv::Mat& input, cv::Mat& output) {
    int rows = input.rows;
    int cols = input.cols;

    output = cv::Mat::zeros(rows, cols, CV_8UC1);

    for (int r = 1; r < rows - 1; ++r) {
        uchar* out_ptr = output.ptr<uchar>(r);
        for (int c = 1; c < cols - 1; ++c) {
            int gx = 0;
            int gy = 0;

            for (int kr = -1; kr <= 1; ++kr) {
                const uchar* in_row = input.ptr<uchar>(r + kr);
                for (int kc = -1; kc <= 1; ++kc) {
                    uchar pixel = in_row[c + kc];
                    gx += pixel * SOBEL_GX[kr + 1][kc + 1];
                    gy += pixel * SOBEL_GY[kr + 1][kc + 1];
                }
            }

            // Magnitude approximation: G = sqrt(Gx^2 + Gy^2)
            int magnitude = static_cast<int>(std::sqrt(gx * gx + gy * gy));
            out_ptr[c] = static_cast<uchar>(std::min(255, magnitude));
        }
    }
}

int main(int argc, char** argv) {
    std::string image_path = (argc > 1) ? argv[1] : "";
    cv::Mat input_raw;

    if (!image_path.empty()) {
        input_raw = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    }

    // Fallback: Generate synthetic matrix if no file is provided
    if (input_raw.empty()) {
        std::cout << "[INFO] No valid input image path provided. Generating 2048x2048 synthetic matrix...\n";
        input_raw.create(2048, 2048, CV_8UC1);
        cv::randu(input_raw, cv::Scalar(0), cv::Scalar(255));
    } else {
        std::cout << "[INFO] Loaded image: " << image_path << " (" << input_raw.cols << "x" << input_raw.rows << ")\n";
    }

    int width = input_raw.cols;
    int height = input_raw.rows;

    // Allocate output matrices upfront
    cv::Mat blurred_img(height, width, CV_8UC1);
    cv::Mat edge_img(height, width, CV_8UC1);

    // Warm-up run (ensures cache warming)
    apply_gaussian_blur(input_raw, blurred_img);
    apply_sobel_edge_detection(blurred_img, edge_img);

    // Timed Execution over multiple iterations
    const int ITERATIONS = 5;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < ITERATIONS; ++i) {
        apply_gaussian_blur(input_raw, blurred_img);
        apply_sobel_edge_detection(blurred_img, edge_img);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    double avg_time_ms = total_time_ms / ITERATIONS;
    double avg_time_sec = avg_time_ms / 1000.0;

    // Performance Calculations
    double total_megapixels = (static_cast<double>(width) * height) / 1e6;
    double throughput = total_megapixels / avg_time_sec;

    std::cout << "\n=========================================\n";
    std::cout << " Sequential Pipeline Performance Metrics\n";
    std::cout << "=========================================\n";
    std::cout << " Image Resolution : " << width << " x " << height << "\n";
    std::cout << " Total Pixels     : " << total_megapixels << " MP\n";
    std::cout << " Average Time (T1): " << avg_time_ms << " ms\n";
    std::cout << " Throughput       : " << throughput << " MP/s\n";
    std::cout << "=========================================\n";

    // Save baseline ground-truth output image
    std::string output_filename = "sequential_reference_output.png";
    if (cv::imwrite(output_filename, edge_img)) {
        std::cout << "[SUCCESS] Saved reference image: " << output_filename << "\n";
    } else {
        std::cerr << "[ERROR] Failed to save reference output image.\n";
    }

    return 0;
}