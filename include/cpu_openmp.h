#ifndef CPU_OPENMP_H
#define CPU_OPENMP_H

#include <opencv2/opencv.hpp>

void process_openmp(const cv::Mat& input, cv::Mat& output, int num_threads);
void process_lee_openmp(const cv::Mat& input, cv::Mat& output, int num_threads);
void segment_water_openmp(
    const cv::Mat& despeckled,
    cv::Mat& water_mask,
    float threshold,
    int num_threads);
void detect_flood_change_openmp(
    const cv::Mat& pre_water_mask,
    const cv::Mat& post_water_mask,
    cv::Mat& new_flood_mask,
    cv::Mat& semantic_map,
    int num_threads);
void cleanup_flood_openmp(
    cv::Mat& flood_mask,
    cv::Mat& semantic_map,
    int minimum_patch_pixels,
    int num_threads);

#endif