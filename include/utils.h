#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double stop_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
};

void log_performance(const std::string& mode, int width, int height, double execution_time_ms, double baseline_time_ms = 0.0);
bool load_image_pair(
    const std::string& pre_path,
    const std::string& post_path,
    cv::Mat& pre_intensity,
    cv::Mat& post_intensity,
    std::string& error);
bool normalize_sar_pair(
    const cv::Mat& pre_intensity,
    const cv::Mat& post_intensity,
    cv::Mat& pre_normalized,
    cv::Mat& post_normalized,
    std::string& error);

struct SpatialMetadata {
    double pixel_size_x_m = 10.0;
    double pixel_size_y_m = 10.0;
    double area_per_pixel_m2 = 100.0;
    double world_origin_x = 0.0;
    double world_origin_y = 0.0;
    double origin_lat_deg = 0.0;
    double origin_lon_deg = 0.0;
    bool has_world_file = false;
    bool world_file_used = false;
    bool has_origin_lat_lon = false;
    std::string status;
};

struct FloodAreaStatistics {
    std::uint64_t total_scene_pixels = 0;
    std::uint64_t permanent_water_pixels = 0;
    std::uint64_t pre_flood_water_pixels = 0;
    std::uint64_t post_flood_water_pixels = 0;
    std::uint64_t new_flood_pixels = 0;
    std::uint64_t receded_water_pixels = 0;
    double total_scene_km2 = 0.0;
    double permanent_water_km2 = 0.0;
    double pre_flood_water_km2 = 0.0;
    double post_flood_water_km2 = 0.0;
    double new_flood_km2 = 0.0;
    double new_flood_hectares = 0.0;
    double new_flood_percentage_of_scene = 0.0;
    double receded_water_km2 = 0.0;
};

struct BenchmarkResult {
    std::string mode;
    std::string pre_image_path;
    std::string post_image_path;
    int width = 0;
    int height = 0;
    int threads = 0;
    double runtime_ms = 0.0;
    double speedup = 0.0;
    double parallel_efficiency_percent = 0.0;
    double throughput_mps = 0.0;
    bool verification_checked = false;
    bool verification_passed = false;
    std::uint64_t mismatched_pixels = 0;
    std::uint64_t semantic_mismatched_pixels = 0;
};

bool load_spatial_metadata(
    const std::string& image_path,
    double configured_pixel_size_m,
    bool pixel_size_overridden,
    SpatialMetadata& metadata,
    std::string& error);
double pixel_count_to_square_kilometers(std::size_t pixel_count, const SpatialMetadata& metadata);
double pixel_count_to_hectares(std::size_t pixel_count, const SpatialMetadata& metadata);
bool compute_flood_area_statistics(
    const cv::Mat& semantic_map,
    const cv::Mat& flood_mask,
    const SpatialMetadata& metadata,
    FloodAreaStatistics& statistics,
    std::string& error);
bool write_flood_report(
    const std::string& output_path,
    const std::string& pre_image_path,
    const std::string& post_image_path,
    int width,
    int height,
    const SpatialMetadata& metadata,
    const FloodAreaStatistics& statistics,
    bool auto_threshold,
    float pre_threshold,
    float post_threshold,
    int minimum_patch_pixels,
    const std::vector<BenchmarkResult>& benchmarks,
    std::string& error);
bool write_flood_csv_report(
    const std::string& output_path,
    const std::string& pre_image_path,
    const std::string& post_image_path,
    int width,
    int height,
    const SpatialMetadata& metadata,
    const FloodAreaStatistics& statistics,
    bool auto_threshold,
    float pre_threshold,
    float post_threshold,
    int minimum_patch_pixels,
    const std::vector<BenchmarkResult>& benchmarks,
    std::string& error);
bool append_benchmark_results(
    const std::string& output_path,
    const std::vector<BenchmarkResult>& benchmarks,
    std::string& error);

#endif