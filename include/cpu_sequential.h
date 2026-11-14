#ifndef CPU_SEQUENTIAL_H
#define CPU_SEQUENTIAL_H

#include <opencv2/opencv.hpp>

void process_sequential(const cv::Mat& input, cv::Mat& output);
void process_lee_sequential(const cv::Mat& input, cv::Mat& output);
void segment_water_sequential(
    const cv::Mat& despeckled,
    cv::Mat& water_mask,
    float threshold);

#endif