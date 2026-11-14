#ifndef SAR_PREPROCESSING_H
#define SAR_PREPROCESSING_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <opencv2/opencv.hpp>

constexpr int kLeeWindowSize = 5;
constexpr float kLeeEquivalentLooks = 4.0f;
constexpr int kOtsuHistogramBins = 256;

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

inline float compute_otsu_threshold(const cv::Mat& input) {
    if (input.empty() || input.type() != CV_32FC1) {
        CV_Error(cv::Error::StsBadArg,
            "compute_otsu_threshold requires a non-empty CV_32FC1 image");
    }
    if (!cv::checkRange(input)) {
        CV_Error(cv::Error::StsBadArg,
            "Otsu threshold input must contain only finite values");
    }

    std::array<std::uint64_t, kOtsuHistogramBins> histogram{};
    std::array<float, kOtsuHistogramBins> bin_max{};
    bin_max.fill(-std::numeric_limits<float>::infinity());
    std::uint64_t total_count = 0;
    double total_sum = 0.0;

    for (int y = 0; y < input.rows; ++y) {
        const float* row = input.ptr<float>(y);
        for (int x = 0; x < input.cols; ++x) {
            const float value = std::clamp(row[x], 0.0f, 1.0f);
            const int bin = std::min(
                kOtsuHistogramBins - 1,
                static_cast<int>(value * kOtsuHistogramBins));
            ++histogram[bin];
            bin_max[bin] = std::max(bin_max[bin], value);
            ++total_count;
            total_sum += bin;
        }
    }

    std::uint64_t background_count = 0;
    double background_sum = 0.0;
    double best_between_class_variance = -1.0;
    int best_threshold_bin = -1;
    for (int bin = 0; bin < kOtsuHistogramBins - 1; ++bin) {
        background_count += histogram[bin];
        background_sum += static_cast<double>(bin) * histogram[bin];
        const std::uint64_t foreground_count = total_count - background_count;
        if (background_count == 0 || foreground_count == 0) {
            continue;
        }

        const double background_mean =
            background_sum / static_cast<double>(background_count);
        const double foreground_mean =
            (total_sum - background_sum) / static_cast<double>(foreground_count);
        const double mean_difference = background_mean - foreground_mean;
        const double between_class_variance =
            static_cast<double>(background_count) *
            static_cast<double>(foreground_count) *
            mean_difference * mean_difference;
        if (between_class_variance > best_between_class_variance) {
            best_between_class_variance = between_class_variance;
            best_threshold_bin = bin;
        }
    }

    if (best_threshold_bin < 0) {
        return 0.0f;
    }

    for (int bin = best_threshold_bin; bin >= 0; --bin) {
        if (histogram[bin] != 0) {
            return bin_max[bin];
        }
    }
    return 0.0f;
}

inline bool is_water_pixel(float backscatter, float threshold) {
    return backscatter <= threshold;
}

#endif
