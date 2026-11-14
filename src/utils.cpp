#include "utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <utility>

namespace {

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool read_world_file(
    const std::filesystem::path& path,
    std::array<double, 6>& values,
    std::string& error) {
    std::ifstream file(path);
    if (!file) {
        error = "Unable to open world file: " + path.string();
        return false;
    }

    for (double& value : values) {
        if (!(file >> value) || !std::isfinite(value)) {
            error = "World file must contain six finite numeric values: " + path.string();
            return false;
        }
    }

    std::string extra;
    if (file >> extra) {
        error = "World file contains more than six values: " + path.string();
        return false;
    }

    const double determinant = values[0] * values[3] - values[1] * values[2];
    if (!std::isfinite(determinant) || std::abs(determinant) <= 1e-12) {
        error = "World file has a zero or invalid pixel-area transform: " + path.string();
        return false;
    }
    return true;
}

std::filesystem::path find_world_file(const std::filesystem::path& image_path) {
    const std::string extension = lowercase(image_path.extension().string());
    std::string conventional_extension;
    if (extension == ".jpg" || extension == ".jpeg") {
        conventional_extension = ".jgw";
    } else if (extension == ".png") {
        conventional_extension = ".pgw";
    } else if (extension == ".tif" || extension == ".tiff") {
        conventional_extension = ".tfw";
    }

    if (!conventional_extension.empty()) {
        const auto conventional_path = image_path.parent_path() /
            (image_path.stem().string() + conventional_extension);
        if (std::filesystem::exists(conventional_path)) {
            return conventional_path;
        }
    }

    const auto generic_path = image_path.parent_path() /
        (image_path.stem().string() + ".wld");
    return std::filesystem::exists(generic_path) ? generic_path : std::filesystem::path();
}

bool projected_crs_uses_meters(const std::filesystem::path& crs_path, std::string& reason) {
    std::ifstream file(crs_path);
    if (!file) {
        reason = "the .prj CRS sidecar is missing or unreadable";
        return false;
    }

    std::ostringstream contents;
    contents << file.rdbuf();
    const std::string wkt = lowercase(contents.str());
    const bool projected =
        wkt.find("projcs[") != std::string::npos ||
        wkt.find("projcrs[") != std::string::npos;
    if (!projected) {
        reason = "the .prj CRS is not a projected coordinate system";
        return false;
    }

    const bool meters =
        wkt.find("unit[\"meter\"") != std::string::npos ||
        wkt.find("unit[\"metre\"") != std::string::npos ||
        wkt.find("lengthunit[\"meter\"") != std::string::npos ||
        wkt.find("lengthunit[\"metre\"") != std::string::npos;
    if (!meters) {
        reason = "the .prj CRS does not declare meter-based projected units";
        return false;
    }
    return true;
}

bool to_float_intensity(const cv::Mat& source, cv::Mat& intensity, const std::string& path, std::string& error) {
    cv::Mat single_channel;
    try {
        switch (source.channels()) {
        case 1:
            single_channel = source;
            break;
        case 3:
            cv::cvtColor(source, single_channel, cv::COLOR_BGR2GRAY);
            break;
        case 4:
            cv::cvtColor(source, single_channel, cv::COLOR_BGRA2GRAY);
            break;
        default:
            error = "Unsupported channel count in image: " + path;
            return false;
        }

        single_channel.convertTo(intensity, CV_32F);
    } catch (const cv::Exception& exception) {
        error = "Unable to convert image to single-channel CV_32F intensity: " +
            path + " (" + exception.what() + ")";
        return false;
    }

    return true;
}

}

bool load_image_pair(
    const std::string& pre_path,
    const std::string& post_path,
    cv::Mat& pre_intensity,
    cv::Mat& post_intensity,
    std::string& error) {
    cv::Mat pre_image = cv::imread(pre_path, cv::IMREAD_UNCHANGED);
    if (pre_image.empty()) {
        error = "Could not load pre-flood image: " + pre_path;
        return false;
    }

    cv::Mat post_image = cv::imread(post_path, cv::IMREAD_UNCHANGED);
    if (post_image.empty()) {
        error = "Could not load post-flood image: " + post_path;
        return false;
    }

    if (pre_image.size() != post_image.size()) {
        error = "Pre- and post-flood images must have matching dimensions; pre-flood is " +
            std::to_string(pre_image.cols) + "x" + std::to_string(pre_image.rows) +
            " and post-flood is " + std::to_string(post_image.cols) + "x" +
            std::to_string(post_image.rows) + ".";
        return false;
    }

    cv::Mat converted_pre;
    cv::Mat converted_post;
    if (!to_float_intensity(pre_image, converted_pre, pre_path, error) ||
        !to_float_intensity(post_image, converted_post, post_path, error)) {
        return false;
    }

    pre_intensity = std::move(converted_pre);
    post_intensity = std::move(converted_post);
    return true;
}

bool load_spatial_metadata(
    const std::string& image_path,
    double configured_pixel_size_m,
    bool pixel_size_overridden,
    SpatialMetadata& metadata,
    std::string& error) {
    const double configured_area_per_pixel =
        configured_pixel_size_m * configured_pixel_size_m;
    if (!std::isfinite(configured_pixel_size_m) || configured_pixel_size_m <= 0.0 ||
        !std::isfinite(configured_area_per_pixel)) {
        error = "Configured pixel size must be a finite positive number of meters.";
        return false;
    }

    metadata = SpatialMetadata{};
    metadata.pixel_size_x_m = configured_pixel_size_m;
    metadata.pixel_size_y_m = configured_pixel_size_m;
    metadata.area_per_pixel_m2 = configured_area_per_pixel;

    const std::filesystem::path source_path(image_path);
    const std::filesystem::path world_path = find_world_file(source_path);
    std::array<double, 6> world_values{};
    std::string world_error;
    if (!world_path.empty()) {
        metadata.has_world_file = true;
        if (!read_world_file(world_path, world_values, world_error)) {
            metadata.status = "World-file calibration ignored (" + world_error +
                "); using " + std::to_string(configured_pixel_size_m) + " m/pixel fallback.";
        } else if (pixel_size_overridden) {
            metadata.world_origin_x = world_values[4];
            metadata.world_origin_y = world_values[5];
            metadata.status = "Using --pixel-size override; world-file scale was not applied.";
        } else {
            const std::filesystem::path crs_path =
                source_path.parent_path() / (source_path.stem().string() + ".prj");
            std::string crs_reason;
            if (projected_crs_uses_meters(crs_path, crs_reason)) {
                const double pixel_x = std::hypot(world_values[0], world_values[1]);
                const double pixel_y = std::hypot(world_values[2], world_values[3]);
                const double pixel_area = std::abs(
                    world_values[0] * world_values[3] -
                    world_values[1] * world_values[2]);
                if (!std::isfinite(pixel_x) || !std::isfinite(pixel_y) ||
                    !std::isfinite(pixel_area) || pixel_x <= 0.0 ||
                    pixel_y <= 0.0 || pixel_area <= 0.0) {
                    metadata.status = "World-file scale is invalid; using " +
                        std::to_string(configured_pixel_size_m) + " m/pixel fallback.";
                } else {
                    metadata.pixel_size_x_m = pixel_x;
                    metadata.pixel_size_y_m = pixel_y;
                    metadata.area_per_pixel_m2 = pixel_area;
                    metadata.world_origin_x = world_values[4];
                    metadata.world_origin_y = world_values[5];
                    metadata.world_file_used = true;
                    metadata.status = "Using world-file pixel scale and projected meter CRS.";
                }
            } else {
                metadata.status = "World-file scale is available but cannot be interpreted safely because " +
                    crs_reason + "; using " +
                    std::to_string(configured_pixel_size_m) + " m/pixel fallback.";
            }
        }
    } else if (pixel_size_overridden) {
        metadata.status = "No world file found; using --pixel-size override.";
    } else {
        metadata.status = "No world file found; using " +
            std::to_string(configured_pixel_size_m) + " m/pixel fallback.";
    }

    return true;
}

double pixel_count_to_square_kilometers(
    std::size_t pixel_count,
    const SpatialMetadata& metadata) {
    return static_cast<double>(pixel_count) * metadata.area_per_pixel_m2 / 1'000'000.0;
}

double pixel_count_to_hectares(
    std::size_t pixel_count,
    const SpatialMetadata& metadata) {
    return static_cast<double>(pixel_count) * metadata.area_per_pixel_m2 / 10'000.0;
}

void log_performance(const std::string& mode, int width, int height, double execution_time_ms, double baseline_time_ms) {
    double total_pixels = static_cast<double>(width) * height;
    double time_sec = execution_time_ms / 1000.0;
    double megapixels_per_sec = (total_pixels / 1e6) / time_sec;

    std::cout << "\n------------------------------------------" << std::endl;
    std::cout << " Execution Mode: " << mode << std::endl;
    std::cout << " Resolution:     " << width << " x " << height << " px" << std::endl;
    std::cout << " Runtime:        " << execution_time_ms << " ms" << std::endl;
    std::cout << " Throughput:     " << megapixels_per_sec << " MP/s" << std::endl;

    if (baseline_time_ms > 0.0) {
        double speedup = baseline_time_ms / execution_time_ms;
        std::cout << " Speedup:        " << speedup << "x" << std::endl;
    }
    std::cout << "------------------------------------------" << std::endl;

    // Log data into CSV
    std::ofstream csv("docs/benchmark_results.csv", std::ios::app);
    csv << mode << "," << width << "x" << height << "," << execution_time_ms << "," << megapixels_per_sec << "\n";
    csv.close();
}