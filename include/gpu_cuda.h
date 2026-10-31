#ifndef GPU_CUDA_H
#define GPU_CUDA_H

#include <opencv2/opencv.hpp>

void process_cuda(const cv::Mat& input, cv::Mat& output);

#endif