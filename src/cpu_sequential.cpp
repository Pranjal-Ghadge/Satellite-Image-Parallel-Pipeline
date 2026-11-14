#include "cpu_sequential.h"
#include <cmath>
#include <algorithm>

void process_sequential(const cv::Mat& input, cv::Mat& output) {
    int rows = input.rows;
    int cols = input.cols;

    cv::Mat blur_temp = cv::Mat::zeros(rows, cols, CV_8UC1);
    output = cv::Mat::zeros(rows, cols, CV_8UC1);

    const int kernel_sum = 273;

    // 1. Optimized Sequential 5x5 Gaussian Blur with Spatial Domain Row Pointers
    for (int y = 2; y < rows - 2; ++y) {
        const uchar* r_m2 = input.ptr<uchar>(y - 2);
        const uchar* r_m1 = input.ptr<uchar>(y - 1);
        const uchar* r_0  = input.ptr<uchar>(y);
        const uchar* r_p1 = input.ptr<uchar>(y + 1);
        const uchar* r_p2 = input.ptr<uchar>(y + 2);
        uchar* out_row = blur_temp.ptr<uchar>(y);

        for (int x = 2; x < cols - 2; ++x) {
            int sum = 
                (r_m2[x - 2] * 1  + r_m2[x - 1] * 4  + r_m2[x] * 7  + r_m2[x + 1] * 4  + r_m2[x + 2] * 1) +
                (r_m1[x - 2] * 4  + r_m1[x - 1] * 16 + r_m1[x] * 26 + r_m1[x + 1] * 16 + r_m1[x + 2] * 4) +
                (r_0 [x - 2] * 7  + r_0 [x - 1] * 26 + r_0 [x] * 41 + r_0 [x + 1] * 26 + r_0 [x + 2] * 7) +
                (r_p1[x - 2] * 4  + r_p1[x - 1] * 16 + r_p1[x] * 26 + r_p1[x + 1] * 16 + r_p1[x + 2] * 4) +
                (r_p2[x - 2] * 1  + r_p2[x - 1] * 4  + r_p2[x] * 7  + r_p2[x + 1] * 4  + r_p2[x + 2] * 1);

            out_row[x] = static_cast<uchar>(sum / kernel_sum);
        }
    }

    // 2. Optimized Sequential 3x3 Sobel Gradient & Edge Magnitude
    for (int y = 1; y < rows - 1; ++y) {
        const uchar* r_top = blur_temp.ptr<uchar>(y - 1);
        const uchar* r_mid = blur_temp.ptr<uchar>(y);
        const uchar* r_bot = blur_temp.ptr<uchar>(y + 1);
        uchar* out_row = output.ptr<uchar>(y);

        for (int x = 1; x < cols - 1; ++x) {
            // Horizontal Gradient Gx: [-1 0 1; -2 0 2; -1 0 1]
            int sumX = (r_top[x + 1] - r_top[x - 1]) +
                       2 * (r_mid[x + 1] - r_mid[x - 1]) +
                       (r_bot[x + 1] - r_bot[x - 1]);

            // Vertical Gradient Gy: [-1 -2 -1; 0 0 0; 1 2 1]
            int sumY = (r_bot[x - 1] - r_top[x - 1]) +
                       2 * (r_bot[x] - r_top[x]) +
                       (r_bot[x + 1] - r_top[x + 1]);

            int mag = static_cast<int>(std::sqrt(sumX * sumX + sumY * sumY));
            out_row[x] = static_cast<uchar>(std::min(255, mag));
        }
    }
}
