#ifndef CPU_OPENMP_H
#define CPU_OPENMP_H

#include <opencv2/opencv.hpp>

void process_openmp(const cv::Mat& input, cv::Mat& output, int num_threads);

#endif