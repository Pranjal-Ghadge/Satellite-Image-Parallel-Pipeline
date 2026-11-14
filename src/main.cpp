#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <limits>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <filesystem>
#include <sstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "sar_preprocessing.h"
#include "cpu_sequential.h"
#include "cpu_openmp.h"
#ifdef USE_CUDA
#include "gpu_cuda.h"
#endif

namespace {

struct FloodPipelineResult {
    cv::Mat pre_normalized;
    cv::Mat post_normalized;
    cv::Mat pre_despeckled;
    cv::Mat post_despeckled;
    cv::Mat pre_water_mask;
    cv::Mat post_water_mask;
    cv::Mat flood_mask;
    cv::Mat semantic_map;
    float pre_threshold = 0.0f;
    float post_threshold = 0.0f;
};

void run_flood_pipeline(
    const cv::Mat& pre_intensity,
    const cv::Mat& post_intensity,
    bool use_openmp,
    int thread_count,
    bool auto_threshold,
    float manual_threshold,
    int minimum_patch_pixels,
    FloodPipelineResult& result) {
    std::string normalization_error;
    if (!normalize_sar_pair(
            pre_intensity, post_intensity,
            result.pre_normalized, result.post_normalized,
            normalization_error)) {
        throw std::runtime_error(normalization_error);
    }

    if (use_openmp) {
        process_lee_openmp(
            result.pre_normalized, result.pre_despeckled, thread_count);
        process_lee_openmp(
            result.post_normalized, result.post_despeckled, thread_count);
    } else {
        process_lee_sequential(
            result.pre_normalized, result.pre_despeckled);
        process_lee_sequential(
            result.post_normalized, result.post_despeckled);
    }

    result.pre_threshold = auto_threshold
        ? compute_otsu_threshold(result.pre_despeckled)
        : manual_threshold;
    result.post_threshold = auto_threshold
        ? compute_otsu_threshold(result.post_despeckled)
        : manual_threshold;

    if (use_openmp) {
        segment_water_openmp(
            result.pre_despeckled, result.pre_water_mask,
            result.pre_threshold, thread_count);
        segment_water_openmp(
            result.post_despeckled, result.post_water_mask,
            result.post_threshold, thread_count);
        detect_flood_change_openmp(
            result.pre_water_mask, result.post_water_mask,
            result.flood_mask, result.semantic_map, thread_count);
        cleanup_flood_openmp(
            result.flood_mask, result.semantic_map,
            minimum_patch_pixels, thread_count);
    } else {
        segment_water_sequential(
            result.pre_despeckled, result.pre_water_mask,
            result.pre_threshold);
        segment_water_sequential(
            result.post_despeckled, result.post_water_mask,
            result.post_threshold);
        detect_flood_change_sequential(
            result.pre_water_mask, result.post_water_mask,
            result.flood_mask, result.semantic_map);
        cleanup_flood_sequential(
            result.flood_mask, result.semantic_map,
            minimum_patch_pixels);
    }
}

std::uint64_t count_mismatched_pixels(
    const cv::Mat& expected,
    const cv::Mat& actual) {
    if (expected.empty() || actual.empty() ||
        expected.type() != CV_8UC1 || actual.type() != CV_8UC1 ||
        expected.size() != actual.size()) {
        throw std::invalid_argument(
            "Equivalence check requires matching, non-empty CV_8UC1 masks.");
    }

    std::uint64_t mismatches = 0;
    for (int y = 0; y < expected.rows; ++y) {
        const std::uint8_t* expected_row =
            expected.ptr<std::uint8_t>(y);
        const std::uint8_t* actual_row =
            actual.ptr<std::uint8_t>(y);
        for (int x = 0; x < expected.cols; ++x) {
            mismatches += expected_row[x] != actual_row[x] ? 1U : 0U;
        }
    }
    return mismatches;
}

double pipeline_throughput_mps(
    int width,
    int height,
    double runtime_ms) {
    const double input_megapixels =
        2.0 * static_cast<double>(width) * height / 1'000'000.0;
    return input_megapixels / (runtime_ms / 1000.0);
}

void print_usage(const char* executable) {
    std::cout
        << "Usage:\n"
        << "  " << executable << " [image_path] [--threads N] [--resolution WIDTHxHEIGHT] "
           "[--pixel-size meters] [--origin-lat deg --origin-lon deg]\n"
        << "  " << executable << " --pre <path> --post <path> [--threads N] "
           "[--benchmark] "
           "[--threshold value | --auto-threshold] "
           "[--min-patch-pixels N] [--pixel-size meters] "
           "[--origin-lat deg --origin-lon deg]\n"
        << "  " << executable << " --help\n\n"
        << "Legacy single-image mode runs the Gaussian+Sobel benchmark.\n"
        << "Paired-image mode normalizes and despeckles both images, segments water, "
           "detects and cleans flood zones, verifies sequential/OpenMP results, "
           "and writes maps and area/benchmark reports.\n"
        << "Options:\n"
        << "  --pre <path>                 Pre-flood image (requires --post)\n"
        << "  --post <path>                Post-flood image (requires --pre)\n"
        << "  --threads N                  Use one OpenMP thread count for this run\n"
        << "  --benchmark                  Sweep OpenMP thread counts 1, 2, 4, 8, 16\n"
        << "  --resolution WIDTHxHEIGHT    Resize legacy single-image input\n"
        << "  --threshold value            Manual dark-water threshold on [0, 1] (default 0.25)\n"
        << "  --auto-threshold             Compute an Otsu threshold independently per image\n"
        << "  --min-patch-pixels N         Minimum retained flood patch area (default 25)\n"
        << "  --pixel-size meters          Pixel size fallback/override in meters (default 10.0)\n"
        << "  --origin-lat deg             Optional origin latitude (-90 to 90)\n"
        << "  --origin-lon deg             Optional origin longitude (-180 to 180)\n"
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

bool parse_finite_double(const std::string& value, double& result) {
    try {
        size_t parsed = 0;
        const double parsed_value = std::stod(value, &parsed);
        if (parsed != value.size() || !std::isfinite(parsed_value)) {
            return false;
        }
        result = parsed_value;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

void print_spatial_summary(
    int width,
    int height,
    const SpatialMetadata& metadata) {
    const std::size_t scene_pixels =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    std::ostringstream summary;
    summary << std::fixed << std::setprecision(3)
            << "Spatial resolution: " << metadata.pixel_size_x_m << " m/pixel (X), "
            << metadata.pixel_size_y_m << " m/pixel (Y)\n"
            << "Area per pixel: " << metadata.area_per_pixel_m2 << " m^2\n"
            << std::setprecision(6)
            << "Total scene area: "
            << pixel_count_to_square_kilometers(scene_pixels, metadata) << " km^2 ("
            << pixel_count_to_hectares(scene_pixels, metadata) << " ha)\n"
            << "Spatial metadata: " << metadata.status << '\n';
    if (metadata.has_origin_lat_lon) {
        summary << std::setprecision(6)
                << "Configured origin: " << metadata.origin_lat_deg << " deg latitude, "
                << metadata.origin_lon_deg << " deg longitude\n";
    }
    std::cout << summary.str();
}

bool create_flood_visualizations(
    const cv::Mat& post_normalized,
    const cv::Mat& semantic_map,
    const cv::Mat& flood_mask,
    cv::Mat& pre_water_visualization,
    cv::Mat& post_water_visualization,
    cv::Mat& flood_overlay,
    std::string& error) {
    if (post_normalized.empty() || post_normalized.type() != CV_32FC1 ||
        semantic_map.empty() || semantic_map.type() != CV_8UC1 ||
        flood_mask.empty() || flood_mask.type() != CV_8UC1 ||
        post_normalized.size() != semantic_map.size() ||
        flood_mask.size() != semantic_map.size()) {
        error = "Flood visualization requires matching normalized image, semantic map, and flood mask.";
        return false;
    }

    pre_water_visualization.create(semantic_map.size(), CV_8UC1);
    post_water_visualization.create(semantic_map.size(), CV_8UC1);
    cv::Mat post_grayscale;
    post_normalized.convertTo(post_grayscale, CV_8UC1, 255.0);
    cv::cvtColor(post_grayscale, flood_overlay, cv::COLOR_GRAY2BGR);

    for (int y = 0; y < semantic_map.rows; ++y) {
        const std::uint8_t* classes = semantic_map.ptr<std::uint8_t>(y);
        const std::uint8_t* flood = flood_mask.ptr<std::uint8_t>(y);
        std::uint8_t* pre_water =
            pre_water_visualization.ptr<std::uint8_t>(y);
        std::uint8_t* post_water =
            post_water_visualization.ptr<std::uint8_t>(y);
        cv::Vec3b* overlay = flood_overlay.ptr<cv::Vec3b>(y);
        for (int x = 0; x < semantic_map.cols; ++x) {
            if (classes[x] > 3 ||
                (classes[x] == 2) != (flood[x] == 255)) {
                error = "Flood visualization found inconsistent flood or semantic classes.";
                return false;
            }
            const bool was_water = classes[x] == 1 || classes[x] == 3;
            const bool is_water = classes[x] == 1 || classes[x] == 2;
            pre_water[x] = was_water ? 255 : 0;
            post_water[x] = is_water ? 255 : 0;
            if (classes[x] == 1 || classes[x] == 2) {
                const cv::Vec3b color = classes[x] == 1
                    ? cv::Vec3b(255, 190, 0)
                    : cv::Vec3b(0, 55, 255);
                constexpr double image_weight = 0.45;
                constexpr double color_weight = 0.55;
                for (int channel = 0; channel < 3; ++channel) {
                    overlay[x][channel] = cv::saturate_cast<std::uint8_t>(
                        image_weight * overlay[x][channel] +
                        color_weight * color[channel]);
                }
            }
        }
    }
    return true;
}

}

int main(int argc, char** argv) {
    std::string input_path = "data/raw/input_1024.jpg";
    std::string pre_path;
    std::string post_path;
    std::vector<int> threads_to_test = {2, 4, 8, 16};
    std::vector<int> flood_threads_to_test = {1, 2, 4, 8, 16};
    bool benchmark_requested = false;
    bool threads_overridden = false;
    int target_width = 0;
    int target_height = 0;
    bool has_positional_input = false;
    double configured_pixel_size_m = 10.0;
    bool pixel_size_overridden = false;
    bool has_origin_lat = false;
    bool has_origin_lon = false;
    double origin_lat_deg = 0.0;
    double origin_lon_deg = 0.0;
    float water_threshold = 0.25f;
    bool threshold_overridden = false;
    bool auto_threshold = false;
    int minimum_patch_pixels = 25;
    bool minimum_patch_overridden = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--auto-threshold") {
            auto_threshold = true;
        } else if (arg == "--benchmark") {
            benchmark_requested = true;
            threads_to_test = {1, 2, 4, 8, 16};
            flood_threads_to_test = {1, 2, 4, 8, 16};
        } else if (arg == "--pre" || arg == "--post" || arg == "--threads" ||
            arg == "--resolution" || arg == "--pixel-size" ||
            arg == "--origin-lat" || arg == "--origin-lon" ||
            arg == "--threshold" || arg == "--min-patch-pixels") {
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
                flood_threads_to_test = {requested_threads};
                threads_overridden = true;
            } else if (arg == "--resolution") {
                const size_t separator = value.find('x');
                if (separator == std::string::npos ||
                    !parse_positive_integer(value.substr(0, separator), target_width) ||
                    !parse_positive_integer(value.substr(separator + 1), target_height)) {
                    std::cerr << "Error: --resolution must be WIDTHxHEIGHT with positive dimensions.\n";
                    return 2;
                }
            } else if (arg == "--pixel-size") {
                if (!parse_finite_double(value, configured_pixel_size_m) ||
                    configured_pixel_size_m <= 0.0) {
                    std::cerr << "Error: --pixel-size must be a finite positive number of meters.\n";
                    return 2;
                }
                pixel_size_overridden = true;
            } else if (arg == "--threshold") {
                double parsed_threshold = 0.0;
                if (!parse_finite_double(value, parsed_threshold) ||
                    parsed_threshold < 0.0 || parsed_threshold > 1.0) {
                    std::cerr << "Error: --threshold must be between 0 and 1.\n";
                    return 2;
                }
                water_threshold = static_cast<float>(parsed_threshold);
                threshold_overridden = true;
            } else if (arg == "--min-patch-pixels") {
                if (!parse_positive_integer(value, minimum_patch_pixels)) {
                    std::cerr << "Error: --min-patch-pixels must be a positive integer.\n";
                    return 2;
                }
                minimum_patch_overridden = true;
            } else if (arg == "--origin-lat") {
                if (!parse_finite_double(value, origin_lat_deg) ||
                    origin_lat_deg < -90.0 || origin_lat_deg > 90.0) {
                    std::cerr << "Error: --origin-lat must be between -90 and 90 degrees.\n";
                    return 2;
                }
                has_origin_lat = true;
            } else {
                if (!parse_finite_double(value, origin_lon_deg) ||
                    origin_lon_deg < -180.0 || origin_lon_deg > 180.0) {
                    std::cerr << "Error: --origin-lon must be between -180 and 180 degrees.\n";
                    return 2;
                }
                has_origin_lon = true;
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
    if (!has_pre_path &&
        (threshold_overridden || auto_threshold || minimum_patch_overridden)) {
        std::cerr << "Error: Water threshold and patch options require paired-image mode (--pre/--post).\n";
        return 2;
    }
    if (has_pre_path && has_positional_input) {
        std::cerr << "Error: Use either a positional image path or --pre/--post, not both.\n";
        return 2;
    }
    if (has_origin_lat != has_origin_lon) {
        std::cerr << "Error: --origin-lat and --origin-lon must be provided together.\n";
        return 2;
    }
    if (auto_threshold && threshold_overridden) {
        std::cerr << "Error: --threshold and --auto-threshold cannot be used together.\n";
        return 2;
    }
    if (benchmark_requested && threads_overridden) {
        std::cerr << "Error: Use either --benchmark or --threads N, not both.\n";
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

        SpatialMetadata metadata;
        if (!load_spatial_metadata(
                pre_path, configured_pixel_size_m, pixel_size_overridden, metadata, error)) {
            std::cerr << "Error: " << error << std::endl;
            return 1;
        }
        if (has_origin_lat) {
            metadata.has_origin_lat_lon = true;
            metadata.origin_lat_deg = origin_lat_deg;
            metadata.origin_lon_deg = origin_lon_deg;
        }

        std::cout << "Validated pre-/post-flood image pair: "
                  << pre_intensity.cols << "x" << pre_intensity.rows
                  << " pixels; both converted to single-channel CV_32F intensity.\n";
        std::cout << "Applied shared linear intensity normalization to [0, 1].\n"
                  << "Applying " << kLeeWindowSize << "x" << kLeeWindowSize
                  << " Lee speckle filter (equivalent looks: "
                  << kLeeEquivalentLooks << ").\n";
        print_spatial_summary(pre_intensity.cols, pre_intensity.rows, metadata);

        std::vector<BenchmarkResult> benchmark_results;
        const auto sequential_start = std::chrono::steady_clock::now();
        FloodPipelineResult sequential_result;
        try {
            run_flood_pipeline(
                pre_intensity, post_intensity, false, 1,
                auto_threshold, water_threshold,
                minimum_patch_pixels, sequential_result);
        } catch (const std::exception& exception) {
            std::cerr << "Error: Sequential flood pipeline failed: "
                      << exception.what() << std::endl;
            return 1;
        }
        const double sequential_time_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - sequential_start).count();

        cv::Mat pre_sequential = sequential_result.pre_despeckled;
        cv::Mat post_sequential = sequential_result.post_despeckled;
        cv::Mat pre_normalized = sequential_result.pre_normalized;
        cv::Mat post_normalized = sequential_result.post_normalized;
        cv::Mat pre_water_mask = sequential_result.pre_water_mask;
        cv::Mat post_water_mask = sequential_result.post_water_mask;
        cv::Mat new_flood_mask = sequential_result.flood_mask;
        cv::Mat semantic_map = sequential_result.semantic_map;
        const float pre_active_threshold = sequential_result.pre_threshold;
        const float post_active_threshold = sequential_result.post_threshold;
        const int image_width = pre_intensity.cols;
        const int image_height = pre_intensity.rows;
        const double sequential_throughput =
            pipeline_throughput_mps(image_width, image_height, sequential_time_ms);
        benchmark_results.push_back({
            "Sequential C++ baseline", pre_path, post_path,
            image_width, image_height, 1, sequential_time_ms,
            1.0, 100.0, sequential_throughput, true, true, 0, 0});

        std::cout << std::fixed << std::setprecision(6)
                  << "Sequential C++ complete flood pipeline: "
                  << sequential_time_ms << " ms; throughput: "
                  << sequential_throughput << " MP/s (both inputs).\n";
        if (auto_threshold) {
            std::cout << "Automatic Otsu thresholds (dark water, <=): pre="
                      << pre_active_threshold << ", post="
                      << post_active_threshold << ".\n";
        } else {
            std::cout << "Manual dark-water threshold (<=): "
                      << water_threshold << ".\n";
        }

        bool all_equivalence_checks_passed = true;
        for (int thread_count : flood_threads_to_test) {
            const auto openmp_start = std::chrono::steady_clock::now();
            FloodPipelineResult openmp_result;
            try {
                run_flood_pipeline(
                    pre_intensity, post_intensity, true, thread_count,
                    auto_threshold, water_threshold,
                    minimum_patch_pixels, openmp_result);
            } catch (const std::exception& exception) {
                std::cerr << "Error: OpenMP flood pipeline failed with "
                          << thread_count << " threads: "
                          << exception.what() << std::endl;
                return 1;
            }
            const double openmp_time_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - openmp_start).count();
            const std::uint64_t flood_mismatches =
                count_mismatched_pixels(new_flood_mask, openmp_result.flood_mask);
            const std::uint64_t semantic_mismatches =
                count_mismatched_pixels(semantic_map, openmp_result.semantic_map);
            const bool verification_passed =
                flood_mismatches == 0 && semantic_mismatches == 0;
            all_equivalence_checks_passed =
                all_equivalence_checks_passed && verification_passed;
            const double speedup = sequential_time_ms / openmp_time_ms;
            const double efficiency =
                speedup / static_cast<double>(thread_count) * 100.0;
            const double throughput =
                pipeline_throughput_mps(image_width, image_height, openmp_time_ms);
            benchmark_results.push_back({
                "OpenMP", pre_path, post_path,
                image_width, image_height, thread_count, openmp_time_ms,
                speedup, efficiency, throughput, true, verification_passed,
                flood_mismatches, semantic_mismatches});

            std::cout << "OpenMP complete flood pipeline (" << thread_count
                      << " threads): " << openmp_time_ms
                      << " ms; speedup: " << speedup
                      << "x; parallel efficiency: " << efficiency
                      << "%; throughput: " << throughput << " MP/s.\n";
            if (verification_passed) {
                std::cout << "Verification: PASSED (0 mismatched pixels)\n";
            } else {
                std::cerr << "Verification: FAILED (" << flood_mismatches
                          << " mismatched flood pixels; "
                          << semantic_mismatches
                          << " mismatched semantic pixels).\n";
            }
        }

        Timer preprocessing_timer;
        Timer cleanup_timer;
#ifdef USE_CUDA
        try {
            cv::Mat pre_cuda;
            cv::Mat post_cuda;
            preprocessing_timer.start();
            process_lee_cuda(pre_normalized, pre_cuda);
            process_lee_cuda(post_normalized, post_cuda);
            const double cuda_time_ms = preprocessing_timer.stop_ms();
            const double pre_difference =
                cv::norm(pre_sequential, pre_cuda, cv::NORM_INF);
            const double post_difference =
                cv::norm(post_sequential, post_cuda, cv::NORM_INF);
            const double max_difference =
                std::max(pre_difference, post_difference);
            std::cout << "CUDA Lee filtering (pre + post): " << cuda_time_ms
                      << " ms; max abs. difference from sequential: "
                      << std::scientific
                      << max_difference
                      << std::fixed << ".\n";
            if (max_difference > 1e-5) {
                std::cerr << "Error: CUDA Lee output differs from sequential beyond tolerance.\n";
                return 1;
            }

            cv::Mat pre_cuda_mask;
            cv::Mat post_cuda_mask;
            segment_water_cuda(
                pre_sequential, pre_cuda_mask, pre_active_threshold);
            segment_water_cuda(
                post_sequential, post_cuda_mask, post_active_threshold);
            if (cv::norm(pre_water_mask, pre_cuda_mask, cv::NORM_INF) != 0.0 ||
                cv::norm(post_water_mask, post_cuda_mask, cv::NORM_INF) != 0.0) {
                std::cerr << "Error: CUDA and sequential water masks do not match exactly.\n";
                return 1;
            }

            cv::Mat cuda_flood_mask;
            cv::Mat cuda_semantic_map;
            detect_flood_change_cuda(
                pre_cuda_mask, post_cuda_mask,
                cuda_flood_mask, cuda_semantic_map);
            cleanup_timer.start();
            cleanup_flood_cuda(
                cuda_flood_mask, cuda_semantic_map, minimum_patch_pixels);
            const double cuda_cleanup_ms = cleanup_timer.stop_ms();
            std::cout << "CUDA morphology + connected-component cleanup: "
                      << cuda_cleanup_ms << " ms.\n";
            if (cv::norm(new_flood_mask, cuda_flood_mask, cv::NORM_INF) != 0.0 ||
                cv::norm(semantic_map, cuda_semantic_map, cv::NORM_INF) != 0.0) {
                std::cerr << "Error: CUDA and sequential flood/class maps do not match exactly.\n";
                return 1;
            }
        } catch (const std::exception& exception) {
            std::cerr << "Error: CUDA flood pipeline failed: "
                      << exception.what() << std::endl;
            return 1;
        }
#endif

        FloodAreaStatistics area_statistics;
        if (!compute_flood_area_statistics(
                semantic_map, new_flood_mask, metadata,
                area_statistics, error)) {
            std::cerr << "Error: " << error << std::endl;
            return 1;
        }

        cv::Mat pre_water_visualization;
        cv::Mat post_water_visualization;
        cv::Mat flood_overlay;
        if (!create_flood_visualizations(
                post_normalized, semantic_map, new_flood_mask,
                pre_water_visualization, post_water_visualization,
                flood_overlay, error)) {
            std::cerr << "Error: " << error << std::endl;
            return 1;
        }

        try {
            std::filesystem::create_directories("data/output");
        } catch (const std::filesystem::filesystem_error& exception) {
            std::cerr << "Error: Unable to create data/output: "
                      << exception.what() << std::endl;
            return 1;
        }
        try {
            if (!cv::imwrite(
                    "data/output/pre_water_mask.png", pre_water_visualization) ||
                !cv::imwrite(
                    "data/output/post_water_mask.png", post_water_visualization) ||
                !cv::imwrite("data/output/flood_mask.png", new_flood_mask) ||
                !cv::imwrite("data/output/new_flood_mask.png", new_flood_mask) ||
                !cv::imwrite("data/output/flood_overlay.png", flood_overlay) ||
                !cv::imwrite("data/output/semantic_classification.png", semantic_map)) {
                std::cerr << "Error: Unable to write flood visualization images to data/output/.\n";
                return 1;
            }
        } catch (const cv::Exception& exception) {
            std::cerr << "Error: Unable to write flood visualization images: "
                      << exception.what() << std::endl;
            return 1;
        }

        if (!write_flood_report(
                "data/output/flood_report.json", pre_path, post_path,
                pre_intensity.cols, pre_intensity.rows, metadata,
                area_statistics, auto_threshold,
                pre_active_threshold, post_active_threshold,
                minimum_patch_pixels, benchmark_results, error)) {
            std::cerr << "Error: " << error << std::endl;
            return 1;
        }
        if (!write_flood_csv_report(
                "data/output/flood_report.csv", pre_path, post_path,
                pre_intensity.cols, pre_intensity.rows, metadata,
                area_statistics, auto_threshold,
                pre_active_threshold, post_active_threshold,
                minimum_patch_pixels, benchmark_results, error)) {
            std::cerr << "Error: " << error << std::endl;
            return 1;
        }
        if (!append_benchmark_results(
                "docs/benchmark_results.csv", benchmark_results, error)) {
            std::cerr << "Error: " << error << std::endl;
            return 1;
        }

        std::cout << std::fixed << std::setprecision(6)
                  << "Flood area statistics (from cleaned semantic map):\n"
                  << "  Total scene: " << area_statistics.total_scene_km2 << " km^2\n"
                  << "  Pre-flood water (permanent + receded): "
                  << area_statistics.pre_flood_water_km2 << " km^2\n"
                  << "  Permanent water: " << area_statistics.permanent_water_km2
                  << " km^2\n"
                  << "  Post-flood total water: "
                  << area_statistics.post_flood_water_km2 << " km^2\n"
                  << "  Newly inundated: " << area_statistics.new_flood_km2
                  << " km^2 (" << area_statistics.new_flood_hectares << " ha, "
                  << area_statistics.new_flood_percentage_of_scene
                  << "% of scene)\n"
                  << "  Receded water: " << area_statistics.receded_water_km2
                  << " km^2\n"
                  << "Water masks matched across active implementations.\n"
                  << "Flood and semantic maps match across all enabled implementations.\n"
                  << "Saved data/output/pre_water_mask.png, "
                     "data/output/post_water_mask.png, data/output/flood_mask.png, "
                     "data/output/flood_overlay.png, and "
                     "data/output/flood_report.json and "
                     "data/output/flood_report.csv.\n";
        return all_equivalence_checks_passed ? 0 : 1;
    }

    cv::Mat input = cv::imread(input_path, cv::IMREAD_GRAYSCALE);

    if (input.empty()) {
        std::cerr << "Error: Could not load image from " << input_path << std::endl;
        std::cerr << "Please place a sample image in data/raw/ directory." << std::endl;
        return -1;
    }

    const int scene_width = input.cols;
    const int scene_height = input.rows;
    SpatialMetadata metadata;
    std::string metadata_error;
    if (!load_spatial_metadata(
            input_path, configured_pixel_size_m, pixel_size_overridden, metadata, metadata_error)) {
        std::cerr << "Error: " << metadata_error << std::endl;
        return 1;
    }
    if (has_origin_lat) {
        metadata.has_origin_lat_lon = true;
        metadata.origin_lat_deg = origin_lat_deg;
        metadata.origin_lon_deg = origin_lon_deg;
    }
    print_spatial_summary(scene_width, scene_height, metadata);

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