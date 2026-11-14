#include "cpu_openmp.h"
#include "sar_preprocessing.h"
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>

void process_openmp(const cv::Mat& input, cv::Mat& output, int num_threads) {
    int rows = input.rows;
    int cols = input.cols;

    cv::Mat blur_temp = cv::Mat::zeros(rows, cols, CV_8UC1);
    output = cv::Mat::zeros(rows, cols, CV_8UC1);

    int gauss[5][5] = {
        {1,  4,  7,  4, 1},
        {4, 16, 26, 16, 4},
        {7, 26, 41, 26, 7},
        {4, 16, 26, 16, 4},
        {1,  4,  7,  4, 1}
    };
    int kernel_sum = 273;

    omp_set_num_threads(num_threads);

    // Parallel Gaussian Filter
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y = 2; y < rows - 2; ++y) {
        for (int x = 2; x < cols - 2; ++x) {
            int sum = 0;
            for (int ky = -2; ky <= 2; ++ky) {
                for (int kx = -2; kx <= 2; ++kx) {
                    sum += input.at<uchar>(y + ky, x + kx) * gauss[ky + 2][kx + 2];
                }

            }
            blur_temp.at<uchar>(y, x) = static_cast<uchar>(sum / kernel_sum);
        }
    }

    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    // Parallel Sobel Edge Detection
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y = 1; y < rows - 1; ++y) {
        for (int x = 1; x < cols - 1; ++x) {
            int sumX = 0, sumY = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    uchar val = blur_temp.at<uchar>(y + ky, x + kx);
                    sumX += val * Gx[ky + 1][kx + 1];
                    sumY += val * Gy[ky + 1][kx + 1];
                }
            }
            int mag = std::sqrt(sumX * sumX + sumY * sumY);
            output.at<uchar>(y, x) = static_cast<uchar>(std::min(255, mag));
        }
    }
}

void process_lee_openmp(const cv::Mat& input, cv::Mat& output, int num_threads) {
    if (input.empty() || input.type() != CV_32FC1) {
        throw std::invalid_argument(
            "process_lee_openmp requires a non-empty CV_32FC1 image");
    }
    if (num_threads <= 0) {
        throw std::invalid_argument("OpenMP thread count must be positive");
    }

    output.create(input.size(), CV_32FC1);
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int y = 0; y < input.rows; ++y) {
        float* output_row = output.ptr<float>(y);
        for (int x = 0; x < input.cols; ++x) {
            output_row[x] = lee_filter_pixel(input, y, x);
        }
    }
}

void segment_water_openmp(
    const cv::Mat& despeckled,
    cv::Mat& water_mask,
    float threshold,
    int num_threads) {
    if (despeckled.empty() || despeckled.type() != CV_32FC1) {
        throw std::invalid_argument(
            "segment_water_openmp requires a non-empty CV_32FC1 image");
    }
    if (!std::isfinite(threshold) || threshold < 0.0f || threshold > 1.0f) {
        throw std::invalid_argument("Water threshold must be between 0 and 1");
    }
    if (num_threads <= 0) {
        throw std::invalid_argument("OpenMP thread count must be positive");
    }

    water_mask.create(despeckled.size(), CV_8UC1);
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int y = 0; y < despeckled.rows; ++y) {
        const float* input_row = despeckled.ptr<float>(y);
        std::uint8_t* mask_row = water_mask.ptr<std::uint8_t>(y);
        for (int x = 0; x < despeckled.cols; ++x) {
            mask_row[x] = is_water_pixel(input_row[x], threshold) ? 255 : 0;
        }
    }
}

void detect_flood_change_openmp(
    const cv::Mat& pre_water_mask,
    const cv::Mat& post_water_mask,
    cv::Mat& new_flood_mask,
    cv::Mat& semantic_map,
    int num_threads) {
    if (!is_binary_water_mask(pre_water_mask) ||
        !is_binary_water_mask(post_water_mask) ||
        pre_water_mask.size() != post_water_mask.size()) {
        throw std::invalid_argument(
            "Flood change detection requires matching binary CV_8UC1 water masks");
    }
    if (num_threads <= 0) {
        throw std::invalid_argument("OpenMP thread count must be positive");
    }

    new_flood_mask.create(pre_water_mask.size(), CV_8UC1);
    semantic_map.create(pre_water_mask.size(), CV_8UC1);
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int y = 0; y < pre_water_mask.rows; ++y) {
        const std::uint8_t* pre_row = pre_water_mask.ptr<std::uint8_t>(y);
        const std::uint8_t* post_row = post_water_mask.ptr<std::uint8_t>(y);
        std::uint8_t* flood_row = new_flood_mask.ptr<std::uint8_t>(y);
        std::uint8_t* semantic_row = semantic_map.ptr<std::uint8_t>(y);
        for (int x = 0; x < pre_water_mask.cols; ++x) {
            const bool was_water = pre_row[x] == 255;
            const bool is_water = post_row[x] == 255;
            flood_row[x] = (!was_water && is_water) ? 255 : 0;
            semantic_row[x] = !was_water
                ? (is_water ? 2 : 0)
                : (is_water ? 1 : 3);
        }
    }
}