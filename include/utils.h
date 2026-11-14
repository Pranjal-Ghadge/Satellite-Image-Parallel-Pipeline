#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <chrono>
#include <cstddef>
#include <string>
#include <fstream>
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

bool load_spatial_metadata(
    const std::string& image_path,
    double configured_pixel_size_m,
    bool pixel_size_overridden,
    SpatialMetadata& metadata,
    std::string& error);
double pixel_count_to_square_kilometers(std::size_t pixel_count, const SpatialMetadata& metadata);
double pixel_count_to_hectares(std::size_t pixel_count, const SpatialMetadata& metadata);

#endif