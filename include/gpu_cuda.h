#ifndef GPU_CUDA_H
#define GPU_CUDA_H

#include <opencv2/opencv.hpp>

void process_cuda(const cv::Mat& input, cv::Mat& output);
void process_lee_cuda(const cv::Mat& input, cv::Mat& output);
void segment_water_cuda(
    const cv::Mat& despeckled,
    cv::Mat& water_mask,
    float threshold);
void detect_flood_change_cuda(
    const cv::Mat& pre_water_mask,
    const cv::Mat& post_water_mask,
    cv::Mat& new_flood_mask,
    cv::Mat& semantic_map);

#endif