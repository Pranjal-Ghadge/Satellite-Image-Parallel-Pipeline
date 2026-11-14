#ifndef CPU_SEQUENTIAL_H
#define CPU_SEQUENTIAL_H

#include <opencv2/opencv.hpp>

void process_sequential(const cv::Mat& input, cv::Mat& output);
void process_lee_sequential(const cv::Mat& input, cv::Mat& output);
void segment_water_sequential(
    const cv::Mat& despeckled,
    cv::Mat& water_mask,
    float threshold);
void detect_flood_change_sequential(
    const cv::Mat& pre_water_mask,
    const cv::Mat& post_water_mask,
    cv::Mat& new_flood_mask,
    cv::Mat& semantic_map);
void cleanup_flood_sequential(
    cv::Mat& flood_mask,
    cv::Mat& semantic_map,
    int minimum_patch_pixels);

#endif