#include "utils.h"
#include "sar_preprocessing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <utility>

namespace {

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string json_escape(const std::string& value) {
    std::ostringstream escaped;
    for (const unsigned char ch : value) {
        switch (ch) {
        case '"':
            escaped << "\\\"";
            break;
        case '\\':
            escaped << "\\\\";
            break;
        case '\b':
            escaped << "\\b";
            break;
        case '\f':
            escaped << "\\f";
            break;
        case '\n':
            escaped << "\\n";
            break;
        case '\r':
            escaped << "\\r";
            break;
        case '\t':
            escaped << "\\t";
            break;
        default:
            if (ch < 0x20) {
                constexpr char hex[] = "0123456789abcdef";
                escaped << "\\u00" << hex[ch >> 4] << hex[ch & 0x0f];
            } else {
                escaped << static_cast<char>(ch);
            }
        }
    }
    return escaped.str();
}

std::string csv_escape(const std::string& value) {
    std::string escaped = "\"";
    for (const char ch : value) {
        if (ch == '"') {
            escaped += "\"\"";
        } else {
            escaped += ch;
        }
    }
    escaped += '"';
    return escaped;
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

bool normalize_sar_pair(
    const cv::Mat& pre_intensity,
    const cv::Mat& post_intensity,
    cv::Mat& pre_normalized,
    cv::Mat& post_normalized,
    std::string& error) {
    if (pre_intensity.empty() || post_intensity.empty() ||
        pre_intensity.type() != CV_32FC1 || post_intensity.type() != CV_32FC1 ||
        pre_intensity.size() != post_intensity.size()) {
        error = "SAR pair normalization requires matching, non-empty CV_32FC1 images.";
        return false;
    }
    if (!cv::checkRange(pre_intensity) || !cv::checkRange(post_intensity)) {
        error = "SAR image intensities must contain only finite values.";
        return false;
    }

    double pre_min = 0.0;
    double pre_max = 0.0;
    double post_min = 0.0;
    double post_max = 0.0;
    cv::minMaxLoc(pre_intensity, &pre_min, &pre_max);
    cv::minMaxLoc(post_intensity, &post_min, &post_max);
    const double shared_min = std::min(pre_min, post_min);
    const double shared_max = std::max(pre_max, post_max);
    const double shared_range = shared_max - shared_min;

    if (!std::isfinite(shared_range)) {
        error = "SAR pair has an invalid intensity range.";
        return false;
    }
    if (shared_range == 0.0) {
        pre_normalized = cv::Mat::zeros(pre_intensity.size(), CV_32FC1);
        post_normalized = cv::Mat::zeros(post_intensity.size(), CV_32FC1);
        return true;
    }

    const double scale = 1.0 / shared_range;
    const double offset = -shared_min * scale;
    pre_intensity.convertTo(pre_normalized, CV_32FC1, scale, offset);
    post_intensity.convertTo(post_normalized, CV_32FC1, scale, offset);
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

bool compute_flood_area_statistics(
    const cv::Mat& semantic_map,
    const cv::Mat& flood_mask,
    const SpatialMetadata& metadata,
    FloodAreaStatistics& statistics,
    std::string& error) {
    if (semantic_map.empty() || semantic_map.type() != CV_8UC1 ||
        flood_mask.empty() || flood_mask.type() != CV_8UC1 ||
        semantic_map.size() != flood_mask.size()) {
        error = "Area quantification requires matching, non-empty CV_8UC1 flood and semantic maps.";
        return false;
    }
    if (!std::isfinite(metadata.area_per_pixel_m2) ||
        metadata.area_per_pixel_m2 <= 0.0) {
        error = "Area quantification requires a finite, positive pixel area.";
        return false;
    }

    FloodAreaStatistics computed;
    computed.total_scene_pixels =
        static_cast<std::uint64_t>(semantic_map.rows) *
        static_cast<std::uint64_t>(semantic_map.cols);
    for (int y = 0; y < semantic_map.rows; ++y) {
        const std::uint8_t* classes = semantic_map.ptr<std::uint8_t>(y);
        const std::uint8_t* flood = flood_mask.ptr<std::uint8_t>(y);
        for (int x = 0; x < semantic_map.cols; ++x) {
            switch (classes[x]) {
            case 0:
                if (flood[x] != 0) {
                    error = "Flood mask and semantic map disagree at a dry-land pixel.";
                    return false;
                }
                break;
            case 1:
                if (flood[x] != 0) {
                    error = "Flood mask and semantic map disagree at a permanent-water pixel.";
                    return false;
                }
                ++computed.permanent_water_pixels;
                break;
            case 2:
                if (flood[x] != 255) {
                    error = "Flood mask and semantic map disagree at a newly inundated pixel.";
                    return false;
                }
                ++computed.new_flood_pixels;
                break;
            case 3:
                if (flood[x] != 0) {
                    error = "Flood mask and semantic map disagree at a receded-water pixel.";
                    return false;
                }
                ++computed.receded_water_pixels;
                break;
            default:
                error = "Semantic map contains an unsupported class value.";
                return false;
            }
        }
    }

    computed.pre_flood_water_pixels =
        computed.permanent_water_pixels + computed.receded_water_pixels;
    computed.post_flood_water_pixels =
        computed.permanent_water_pixels + computed.new_flood_pixels;
    computed.total_scene_km2 = pixel_count_to_square_kilometers(
        static_cast<std::size_t>(computed.total_scene_pixels), metadata);
    computed.permanent_water_km2 = pixel_count_to_square_kilometers(
        static_cast<std::size_t>(computed.permanent_water_pixels), metadata);
    computed.pre_flood_water_km2 = pixel_count_to_square_kilometers(
        static_cast<std::size_t>(computed.pre_flood_water_pixels), metadata);
    computed.post_flood_water_km2 = pixel_count_to_square_kilometers(
        static_cast<std::size_t>(computed.post_flood_water_pixels), metadata);
    computed.new_flood_km2 = pixel_count_to_square_kilometers(
        static_cast<std::size_t>(computed.new_flood_pixels), metadata);
    computed.new_flood_hectares = pixel_count_to_hectares(
        static_cast<std::size_t>(computed.new_flood_pixels), metadata);
    computed.receded_water_km2 = pixel_count_to_square_kilometers(
        static_cast<std::size_t>(computed.receded_water_pixels), metadata);
    computed.new_flood_percentage_of_scene =
        computed.total_scene_pixels == 0
        ? 0.0
        : 100.0 * static_cast<double>(computed.new_flood_pixels) /
            static_cast<double>(computed.total_scene_pixels);
    statistics = computed;
    return true;
}

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
    std::string& error) {
    if (width <= 0 || height <= 0 || minimum_patch_pixels <= 0 ||
        !std::isfinite(metadata.pixel_size_x_m) ||
        !std::isfinite(metadata.pixel_size_y_m) ||
        !std::isfinite(pre_threshold) || !std::isfinite(post_threshold)) {
        error = "Cannot write flood report: dimensions or processing parameters are invalid.";
        return false;
    }

    std::ofstream report(output_path, std::ios::out | std::ios::trunc);
    if (!report) {
        error = "Unable to open flood report for writing: " + output_path;
        return false;
    }

    report << std::setprecision(15)
        << "{\n"
        << "  \"images\": {\n"
        << "    \"pre_flood\": \"" << json_escape(pre_image_path) << "\",\n"
        << "    \"post_flood\": \"" << json_escape(post_image_path) << "\",\n"
        << "    \"width_pixels\": " << width << ",\n"
        << "    \"height_pixels\": " << height << "\n"
        << "  },\n"
        << "  \"spatial_resolution\": {\n"
        << "    \"pixel_size_x_m\": " << metadata.pixel_size_x_m << ",\n"
        << "    \"pixel_size_y_m\": " << metadata.pixel_size_y_m << ",\n"
        << "    \"area_per_pixel_m2\": " << metadata.area_per_pixel_m2 << ",\n"
        << "    \"metadata_status\": \"" << json_escape(metadata.status) << "\"\n"
        << "  },\n"
        << "  \"bounding_coordinates\": ";
    if (metadata.has_origin_lat_lon || metadata.world_file_used) {
        report << "{\n"
            << "    \"geographic_origin\": ";
        if (metadata.has_origin_lat_lon) {
            report << "{ \"latitude_deg\": " << metadata.origin_lat_deg
                   << ", \"longitude_deg\": " << metadata.origin_lon_deg << " }";
        } else {
            report << "null";
        }
        report << ",\n"
            << "    \"projected_world_file_origin_m\": ";
        if (metadata.world_file_used) {
            report << "{ \"x\": " << metadata.world_origin_x
                   << ", \"y\": " << metadata.world_origin_y << " }";
        } else {
            report << "null";
        }
        report << ",\n"
            << "    \"full_extent\": null\n"
            << "  },\n";
    } else {
        report << "null,\n";
    }
    report
        << "  \"pixel_counts\": {\n"
        << "    \"total_scene\": " << statistics.total_scene_pixels << ",\n"
        << "    \"permanent_water\": " << statistics.permanent_water_pixels << ",\n"
        << "    \"pre_flood_water\": " << statistics.pre_flood_water_pixels << ",\n"
        << "    \"post_flood_total_water\": " << statistics.post_flood_water_pixels << ",\n"
        << "    \"newly_inundated_flood\": " << statistics.new_flood_pixels << ",\n"
        << "    \"receded_water\": " << statistics.receded_water_pixels << "\n"
        << "  },\n"
        << "  \"areas\": {\n"
        << "    \"total_scene_km2\": " << statistics.total_scene_km2 << ",\n"
        << "    \"permanent_water_km2\": " << statistics.permanent_water_km2 << ",\n"
        << "    \"pre_flood_water_km2\": " << statistics.pre_flood_water_km2 << ",\n"
        << "    \"post_flood_total_water_km2\": " << statistics.post_flood_water_km2 << ",\n"
        << "    \"newly_inundated_flood_km2\": " << statistics.new_flood_km2 << ",\n"
        << "    \"newly_inundated_flood_hectares\": " << statistics.new_flood_hectares << ",\n"
        << "    \"receded_water_km2\": " << statistics.receded_water_km2 << ",\n"
        << "    \"newly_inundated_flood_percent_of_scene\": "
        << statistics.new_flood_percentage_of_scene << "\n"
        << "  },\n"
        << "  \"processing_parameters\": {\n"
        << "    \"normalization\": \"shared linear min-max [0, 1]\",\n"
        << "    \"lee_window_size\": " << kLeeWindowSize << ",\n"
        << "    \"lee_equivalent_looks\": " << kLeeEquivalentLooks << ",\n"
        << "    \"threshold_mode\": \"" << (auto_threshold ? "otsu" : "manual") << "\",\n"
        << "    \"pre_water_threshold\": " << pre_threshold << ",\n"
        << "    \"post_water_threshold\": " << post_threshold << ",\n"
        << "    \"minimum_patch_pixels\": " << minimum_patch_pixels << ",\n"
        << "    \"cleanup\": \"3x3 opening, 3x3 closing, 8-connected component filter\"\n"
        << "  },\n"
        << "  \"benchmark\": {\n";
    bool has_baseline = false;
    for (const BenchmarkResult& benchmark : benchmarks) {
        if (benchmark.mode == "Sequential C++ baseline") {
            report << "    \"sequential_runtime_ms\": " << benchmark.runtime_ms << ",\n"
                << "    \"sequential_throughput_mps\": "
                << benchmark.throughput_mps << ",\n";
            has_baseline = true;
            break;
        }
    }
    if (!has_baseline) {
        report << "    \"sequential_runtime_ms\": null,\n"
            << "    \"sequential_throughput_mps\": null,\n";
    }
    report << "    \"openmp_runs\": [";
    bool first_openmp_run = true;
    for (const BenchmarkResult& benchmark : benchmarks) {
        if (benchmark.mode != "OpenMP") {
            continue;
        }
        report << (first_openmp_run ? "\n" : ",\n")
            << "      { \"threads\": " << benchmark.threads
            << ", \"runtime_ms\": " << benchmark.runtime_ms
            << ", \"speedup\": " << benchmark.speedup
            << ", \"parallel_efficiency_percent\": "
            << benchmark.parallel_efficiency_percent
            << ", \"throughput_mps\": " << benchmark.throughput_mps
            << ", \"verification_passed\": "
            << (benchmark.verification_passed ? "true" : "false")
            << ", \"mismatched_pixels\": " << benchmark.mismatched_pixels
            << ", \"semantic_mismatched_pixels\": "
            << benchmark.semantic_mismatched_pixels
            << " }";
        first_openmp_run = false;
    }
    if (!first_openmp_run) {
        report << '\n';
    }
    report << "    ],\n"
        << "    \"timed_stages\": \"shared pair normalization, Lee despeckling, threshold selection, water segmentation, change detection, and cleanup\",\n"
        << "    \"throughput_definition\": \"both input images in megapixels per second\"\n"
        << "  }\n"
        << "}\n";
    if (!report) {
        error = "Failed while writing flood report: " + output_path;
        return false;
    }
    return true;
}

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
    std::string& error) {
    if (width <= 0 || height <= 0 || minimum_patch_pixels <= 0 ||
        !std::isfinite(metadata.pixel_size_x_m) ||
        !std::isfinite(metadata.pixel_size_y_m) ||
        !std::isfinite(pre_threshold) || !std::isfinite(post_threshold)) {
        error = "Cannot write flood CSV report: dimensions or processing parameters are invalid.";
        return false;
    }

    std::ofstream report(output_path, std::ios::out | std::ios::trunc);
    if (!report) {
        error = "Unable to open flood CSV report for writing: " + output_path;
        return false;
    }

    report << "section,key,value,unit\n"
        << "images,pre_flood_path," << csv_escape(pre_image_path) << ",\n"
        << "images,post_flood_path," << csv_escape(post_image_path) << ",\n"
        << "images,width," << width << ",pixels\n"
        << "images,height," << height << ",pixels\n"
        << "spatial_resolution,pixel_size_x," << std::setprecision(15)
        << metadata.pixel_size_x_m << ",m/pixel\n"
        << "spatial_resolution,pixel_size_y," << metadata.pixel_size_y_m << ",m/pixel\n"
        << "spatial_resolution,area_per_pixel," << metadata.area_per_pixel_m2 << ",m2\n"
        << "spatial_resolution,metadata_status,"
        << csv_escape(metadata.status) << ",\n";
    if (metadata.has_origin_lat_lon) {
        report << "bounding_coordinates,origin_latitude,"
               << metadata.origin_lat_deg << ",degrees\n"
               << "bounding_coordinates,origin_longitude,"
               << metadata.origin_lon_deg << ",degrees\n";
    }
    if (metadata.world_file_used) {
        report << "bounding_coordinates,world_file_origin_x,"
               << metadata.world_origin_x << ",projected_m\n"
               << "bounding_coordinates,world_file_origin_y,"
               << metadata.world_origin_y << ",projected_m\n";
    }
    report << "pixel_counts,total_scene," << statistics.total_scene_pixels << ",pixels\n"
        << "pixel_counts,permanent_water," << statistics.permanent_water_pixels << ",pixels\n"
        << "pixel_counts,pre_flood_water," << statistics.pre_flood_water_pixels << ",pixels\n"
        << "pixel_counts,post_flood_total_water," << statistics.post_flood_water_pixels << ",pixels\n"
        << "pixel_counts,newly_inundated_flood," << statistics.new_flood_pixels << ",pixels\n"
        << "pixel_counts,receded_water," << statistics.receded_water_pixels << ",pixels\n"
        << "areas,total_scene," << statistics.total_scene_km2 << ",km2\n"
        << "areas,permanent_water," << statistics.permanent_water_km2 << ",km2\n"
        << "areas,pre_flood_water," << statistics.pre_flood_water_km2 << ",km2\n"
        << "areas,post_flood_total_water," << statistics.post_flood_water_km2 << ",km2\n"
        << "areas,newly_inundated_flood," << statistics.new_flood_km2 << ",km2\n"
        << "areas,newly_inundated_flood," << statistics.new_flood_hectares << ",ha\n"
        << "areas,newly_inundated_flood_percent_of_scene,"
        << statistics.new_flood_percentage_of_scene << ",percent\n"
        << "areas,receded_water," << statistics.receded_water_km2 << ",km2\n"
        << "processing_parameters,threshold_mode,"
        << (auto_threshold ? "otsu" : "manual") << ",\n"
        << "processing_parameters,pre_water_threshold," << pre_threshold << ",normalized\n"
        << "processing_parameters,post_water_threshold," << post_threshold << ",normalized\n"
        << "processing_parameters,minimum_patch_pixels,"
        << minimum_patch_pixels << ",pixels\n"
        << "processing_parameters,lee_window_size," << kLeeWindowSize << ",pixels\n"
        << "processing_parameters,lee_equivalent_looks,"
        << kLeeEquivalentLooks << ",\n"
        << "processing_parameters,cleanup,"
        << csv_escape("3x3 opening, 3x3 closing, 8-connected component filter") << ",\n";
    for (const BenchmarkResult& benchmark : benchmarks) {
        report << "benchmark," << csv_escape(benchmark.mode) << ","
            << benchmark.runtime_ms << ",ms\n"
            << "benchmark,threads," << benchmark.threads << ",threads\n"
            << "benchmark,speedup," << benchmark.speedup << ",x\n"
            << "benchmark,parallel_efficiency,"
            << benchmark.parallel_efficiency_percent << ",percent\n"
            << "benchmark,throughput," << benchmark.throughput_mps << ",MP/s\n"
            << "benchmark,verification,"
            << (benchmark.verification_passed ? "PASSED" : "FAILED") << ",\n"
            << "benchmark,mismatched_pixels,"
            << benchmark.mismatched_pixels << ",pixels\n"
            << "benchmark,semantic_mismatched_pixels,"
            << benchmark.semantic_mismatched_pixels << ",pixels\n";
    }
    if (!report) {
        error = "Failed while writing flood CSV report: " + output_path;
        return false;
    }
    return true;
}

bool append_benchmark_results(
    const std::string& output_path,
    const std::vector<BenchmarkResult>& benchmarks,
    std::string& error) {
    for (const BenchmarkResult& benchmark : benchmarks) {
        if (benchmark.width <= 0 || benchmark.height <= 0 ||
            !std::isfinite(benchmark.runtime_ms) || benchmark.runtime_ms <= 0.0 ||
            !std::isfinite(benchmark.throughput_mps) ||
            !std::isfinite(benchmark.speedup) ||
            !std::isfinite(benchmark.parallel_efficiency_percent) ||
            benchmark.threads < 0) {
            error = "Cannot append invalid benchmark metrics.";
            return false;
        }
    }

    const std::string header =
        "mode,resolution,runtime_ms,throughput_mps,threads,speedup,"
        "parallel_efficiency_percent,verification,mismatched_flood_pixels,"
        "mismatched_semantic_pixels";
    const auto write_benchmark_rows = [&](std::ostream& csv) {
        csv << std::setprecision(15);
        for (const BenchmarkResult& benchmark : benchmarks) {
            csv << csv_escape(benchmark.mode) << ","
                << benchmark.width << "x" << benchmark.height << ","
                << benchmark.runtime_ms << ","
                << benchmark.throughput_mps << ","
                << benchmark.threads << ","
                << benchmark.speedup << ","
                << benchmark.parallel_efficiency_percent << ","
                << (benchmark.verification_checked
                    ? (benchmark.verification_passed ? "PASSED" : "FAILED")
                    : "N/A") << ","
                << (benchmark.verification_checked
                    ? std::to_string(benchmark.mismatched_pixels)
                    : std::string()) << ","
                << (benchmark.verification_checked
                    ? std::to_string(benchmark.semantic_mismatched_pixels)
                    : std::string()) << "\n";
        }
    };

    const std::filesystem::path csv_path(output_path);
    std::error_code filesystem_error;
    if (!csv_path.parent_path().empty()) {
        std::filesystem::create_directories(
            csv_path.parent_path(), filesystem_error);
        if (filesystem_error) {
            error = "Unable to create benchmark CSV directory: " +
                filesystem_error.message();
            return false;
        }
    }

    std::ifstream existing_file(output_path);
    if (!existing_file) {
        std::ofstream csv(output_path, std::ios::out | std::ios::trunc);
        if (!csv) {
            error = "Unable to create benchmark results CSV: " + output_path;
            return false;
        }
        csv << header << '\n';
        write_benchmark_rows(csv);
        if (!csv) {
            error = "Failed while writing benchmark results CSV: " + output_path;
            return false;
        }
        return true;
    }

    std::vector<std::string> existing_lines;
    std::string line;
    while (std::getline(existing_file, line)) {
        if (!line.empty()) {
            existing_lines.push_back(line);
        }
    }
    if (existing_file.bad()) {
        error = "Unable to read existing benchmark results CSV: " + output_path;
        return false;
    }
    existing_file.close();
    if (!existing_lines.empty() && existing_lines.front() == header) {
        std::ofstream csv(output_path, std::ios::out | std::ios::app);
        if (!csv) {
            error = "Unable to open benchmark results CSV for appending: " + output_path;
            return false;
        }
        write_benchmark_rows(csv);
        if (!csv) {
            error = "Failed while appending benchmark results CSV: " + output_path;
            return false;
        }
        return true;
    }

    std::vector<std::string> migrated_rows;
    migrated_rows.reserve(existing_lines.size());
    for (const std::string& existing_line : existing_lines) {
        std::istringstream row_stream(existing_line);
        std::vector<std::string> fields;
        std::string field;
        while (std::getline(row_stream, field, ',')) {
            fields.push_back(field);
        }
        if (fields.size() != 4) {
            error = "Existing benchmark CSV has an unsupported row; refusing to rewrite it.";
            return false;
        }
        const std::size_t separator = fields[1].find('x');
        if (separator == std::string::npos) {
            error = "Existing benchmark CSV has an invalid resolution; refusing to rewrite it.";
            return false;
        }
        migrated_rows.push_back(
            csv_escape(fields[0]) + "," + fields[1] + "," +
            fields[2] + "," + fields[3] + ",,,,,,\n");
    }

    const std::filesystem::path temporary_path =
        csv_path.string() + ".phase8.tmp";
    const std::filesystem::path backup_path =
        csv_path.string() + ".phase8.bak";
    if (std::filesystem::exists(temporary_path) ||
        std::filesystem::exists(backup_path)) {
        error = "Temporary benchmark CSV migration files already exist; refusing to overwrite them.";
        return false;
    }

    {
        std::ofstream temporary(temporary_path, std::ios::out | std::ios::trunc);
        if (!temporary) {
            error = "Unable to create temporary benchmark CSV: " +
                temporary_path.string();
            return false;
        }
        temporary << header << '\n';
        for (const std::string& migrated_row : migrated_rows) {
            temporary << migrated_row;
        }
        write_benchmark_rows(temporary);
        if (!temporary) {
            error = "Failed while creating migrated benchmark CSV.";
            temporary.close();
            std::filesystem::remove(temporary_path, filesystem_error);
            return false;
        }
    }

    std::filesystem::rename(csv_path, backup_path, filesystem_error);
    if (filesystem_error) {
        error = "Unable to preserve the existing benchmark CSV during migration: " +
            filesystem_error.message();
        std::filesystem::remove(temporary_path, filesystem_error);
        return false;
    }
    std::filesystem::rename(temporary_path, csv_path, filesystem_error);
    if (filesystem_error) {
        const std::string rename_error = filesystem_error.message();
        std::error_code restore_error;
        std::filesystem::rename(backup_path, csv_path, restore_error);
        std::filesystem::remove(temporary_path, restore_error);
        error = "Unable to install migrated benchmark CSV: " + rename_error;
        return false;
    }
    std::filesystem::remove(backup_path, filesystem_error);
    if (filesystem_error) {
        error = "Benchmark CSV was updated, but its migration backup could not be removed: " +
            filesystem_error.message();
        return false;
    }
    return true;
}

void log_performance(const std::string& mode, int width, int height, double execution_time_ms, double baseline_time_ms) {
    const double total_pixels = static_cast<double>(width) * height;
    const double time_sec = execution_time_ms / 1000.0;
    const double megapixels_per_sec = (total_pixels / 1e6) / time_sec;

    std::cout << "\n------------------------------------------" << std::endl;
    std::cout << " Execution Mode: " << mode << std::endl;
    std::cout << " Resolution:     " << width << " x " << height << " px" << std::endl;
    std::cout << " Runtime:        " << execution_time_ms << " ms" << std::endl;
    std::cout << " Throughput:     " << megapixels_per_sec << " MP/s" << std::endl;

    if (baseline_time_ms > 0.0) {
        const double speedup = baseline_time_ms / execution_time_ms;
        std::cout << " Speedup:        " << speedup << "x" << std::endl;
    }
    std::cout << "------------------------------------------" << std::endl;

    int threads = 0;
    const std::size_t opening = mode.find('(');
    if (opening != std::string::npos) {
        std::size_t digit = opening + 1;
        while (digit < mode.size() &&
            (mode[digit] < '0' || mode[digit] > '9')) {
            ++digit;
        }
        if (digit < mode.size()) {
            try {
                threads = std::stoi(mode.substr(digit));
            } catch (const std::exception&) {
                threads = 0;
            }
        }
    }

    const double speedup = baseline_time_ms > 0.0
        ? baseline_time_ms / execution_time_ms
        : 1.0;
    const double efficiency = threads > 0
        ? speedup / static_cast<double>(threads) * 100.0
        : 0.0;
    BenchmarkResult result;
    result.mode = mode;
    result.width = width;
    result.height = height;
    result.threads = threads;
    result.runtime_ms = execution_time_ms;
    result.speedup = speedup;
    result.parallel_efficiency_percent = efficiency;
    result.throughput_mps = megapixels_per_sec;
    std::string error;
    if (!append_benchmark_results(
            "docs/benchmark_results.csv", {result}, error)) {
        std::cerr << "Error: " << error << std::endl;
    }
}