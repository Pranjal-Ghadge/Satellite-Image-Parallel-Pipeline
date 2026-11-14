#ifndef SAR_PREPROCESSING_H
#define SAR_PREPROCESSING_H

#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

constexpr int kLeeWindowSize = 5;
constexpr float kLeeEquivalentLooks = 4.0f;

inline float lee_filter_pixel(const cv::Mat& input, int y, int x) {
    constexpr int radius = kLeeWindowSize / 2;
    constexpr float epsilon = 1e-12f;
    float sum = 0.0f;
    float sum_squares = 0.0f;

    for (int ky = -radius; ky <= radius; ++ky) {
        const int sample_y = std::max(0, std::min(input.rows - 1, y + ky));
        const float* row = input.ptr<float>(sample_y);
        for (int kx = -radius; kx <= radius; ++kx) {
            const int sample_x = std::max(0, std::min(input.cols - 1, x + kx));
            const float value = row[sample_x];
            sum += value;
            sum_squares += value * value;
        }
    }

    constexpr float sample_count =
        static_cast<float>(kLeeWindowSize * kLeeWindowSize);
    const float local_mean = sum / sample_count;
    const float local_variance =
        std::max(0.0f, sum_squares / sample_count - local_mean * local_mean);
    const float noise_variance =
        local_mean * local_mean / kLeeEquivalentLooks;
    const float weight = std::max(
        0.0f,
        (local_variance - noise_variance) / (local_variance + epsilon));
    const float center = input.ptr<float>(y)[x];
    return local_mean + weight * (center - local_mean);
}

#endif
