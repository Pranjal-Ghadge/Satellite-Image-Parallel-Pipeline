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

#endif