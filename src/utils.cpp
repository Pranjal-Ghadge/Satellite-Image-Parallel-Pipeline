#include "utils.h"

#include <utility>

namespace {

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