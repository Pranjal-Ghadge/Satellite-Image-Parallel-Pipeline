#ifndef SAR_PREPROCESSING_H
#define SAR_PREPROCESSING_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <opencv2/opencv.hpp>
#include <vector>

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

inline bool is_binary_water_mask(const cv::Mat& mask) {
    if (mask.empty() || mask.type() != CV_8UC1) {
        return false;
    }

    for (int y = 0; y < mask.rows; ++y) {
        const std::uint8_t* row = mask.ptr<std::uint8_t>(y);
        for (int x = 0; x < mask.cols; ++x) {
            if (row[x] != 0 && row[x] != 255) {
                return false;
            }
        }
    }
    return true;
}

inline void cleanup_flood_classification(
    cv::Mat& flood_mask,
    cv::Mat& semantic_map,
    int minimum_patch_pixels) {
    if (!is_binary_water_mask(flood_mask) ||
        semantic_map.empty() || semantic_map.type() != CV_8UC1 ||
        semantic_map.size() != flood_mask.size()) {
        CV_Error(cv::Error::StsBadArg,
            "Flood cleanup requires a binary CV_8UC1 mask and a matching CV_8UC1 semantic map");
    }
    if (minimum_patch_pixels <= 0) {
        CV_Error(cv::Error::StsBadArg,
            "Minimum flood patch area must be a positive number of pixels");
    }

    for (int y = 0; y < semantic_map.rows; ++y) {
        const std::uint8_t* row = semantic_map.ptr<std::uint8_t>(y);
        for (int x = 0; x < semantic_map.cols; ++x) {
            if (row[x] > 3) {
                CV_Error(cv::Error::StsBadArg,
                    "Semantic map values must be between 0 and 3");
            }
        }
    }

    const cv::Mat original_flood_mask = flood_mask.clone();
    cv::Mat cleaned;
    const cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(
        original_flood_mask, cleaned, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(cleaned, cleaned, cv::MORPH_CLOSE, kernel);

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(
        cleaned, labels, stats, centroids, 8, CV_32S);

    std::vector<std::uint8_t> keep_component(
        static_cast<std::size_t>(component_count), 0);
    for (int label = 1; label < component_count; ++label) {
        if (stats.at<int>(label, cv::CC_STAT_AREA) >= minimum_patch_pixels) {
            keep_component[static_cast<std::size_t>(label)] = 1;
        }
    }

    for (int y = 0; y < flood_mask.rows; ++y) {
        const int* label_row = labels.ptr<int>(y);
        const std::uint8_t* original_row =
            original_flood_mask.ptr<std::uint8_t>(y);
        std::uint8_t* flood_row = flood_mask.ptr<std::uint8_t>(y);
        std::uint8_t* semantic_row = semantic_map.ptr<std::uint8_t>(y);
        for (int x = 0; x < flood_mask.cols; ++x) {
            const bool retained =
                keep_component[static_cast<std::size_t>(label_row[x])] != 0;
            flood_row[x] = retained ? 255 : 0;
            if (retained) {
                semantic_row[x] = 2;
            } else if (original_row[x] == 255 && semantic_row[x] == 2) {
                semantic_row[x] = 0;
            }
        }
    }
}

#endif
