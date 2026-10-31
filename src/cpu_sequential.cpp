#include "cpu_sequential.h"
#include <cmath>
#include <algorithm>

void process_sequential(const cv::Mat& input, cv::Mat& output) {
    int rows = input.rows;
    int cols = input.cols;
    
    cv::Mat blur_temp = cv::Mat::zeros(rows, cols, CV_8UC1);
    output = cv::Mat::zeros(rows, cols, CV_8UC1);

    // 5x5 Gaussian Kernel
    int gauss[5][5] = {
        {1,  4,  7,  4, 1},
        {4, 16, 26, 16, 4},
        {7, 26, 41, 26, 7},
        {4, 16, 26, 16, 4},
        {1,  4,  7,  4, 1}
    };
    int kernel_sum = 273;

    // 1. Gaussian Blur Stage
    for (int y = 2; y < rows - 2; ++y) {
        for (int x = 2; x < cols - 2; ++x) {
            int sum = 0;
            for (int ky = -2; ky <= 2; ++ky) {
                for (int kx = -2; kx <= 2; ++kx) {
                    sum += input.at<uchar>(y + ky, x + kx) * gauss[ky + 2][kx + 2];
                }
            }
            blur_temp.at<uchar>(y, x) = static_cast<uchar>(sum / kernel_sum);
        }
    }

    // 3x3 Sobel Gradients
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    // 2. Sobel Edge Detection Stage
    for (int y = 1; y < rows - 1; ++y) {
        for (int x = 1; x < cols - 1; ++x) {
            int sumX = 0, sumY = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    uchar val = blur_temp.at<uchar>(y + ky, x + kx);
                    sumX += val * Gx[ky + 1][kx + 1];
                    sumY += val * Gy[ky + 1][kx + 1];
                }
            }
            int mag = std::sqrt(sumX * sumX + sumY * sumY);
            output.at<uchar>(y, x) = static_cast<uchar>(std::min(255, mag));
        }
    }
}