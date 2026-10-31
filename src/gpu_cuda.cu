#include "gpu_cuda.h"

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <cstring>
#include <iostream>
#include <algorithm>
#include <stdexcept>

__global__ void gaussian_blur_kernel(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ blurred,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 2 && x < width - 2 && y >= 2 && y < height - 2) {
        const int gauss[5][5] = {
            {1,  4,  7,  4, 1},
            {4, 16, 26, 16, 4},
            {7, 26, 41, 26, 7},
            {4, 16, 26, 16, 4},
            {1,  4,  7,  4, 1}
        };

        int g_sum = 0;
        for (int ky = -2; ky <= 2; ++ky) {
            for (int kx = -2; kx <= 2; ++kx) {
                g_sum += input[(y + ky) * width + (x + kx)] * gauss[ky + 2][kx + 2];
            }
        }
        blurred[y * width + x] = static_cast<unsigned char>(g_sum / 273);
    }
}

__global__ void sobel_kernel(
    const unsigned char* __restrict__ blurred,
    unsigned char* __restrict__ output,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        const int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

        int sumX = 0, sumY = 0;
        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                unsigned char p = blurred[(y + ky) * width + (x + kx)];
                sumX += p * Gx[ky + 1][kx + 1];
                sumY += p * Gy[ky + 1][kx + 1];
            }
        }

        int mag = sqrtf(static_cast<float>(sumX * sumX + sumY * sumY));
        output[y * width + x] = static_cast<unsigned char>(mag > 255 ? 255 : mag);
    }
}

void process_cuda(const cv::Mat& input, cv::Mat& output) {
    int width = input.cols;
    int height = input.rows;
    size_t img_bytes = width * height * sizeof(unsigned char);

    if (input.empty() || input.type() != CV_8UC1) {
        throw std::invalid_argument("process_cuda requires a non-empty 8-bit grayscale cv::Mat");
    }

    output = cv::Mat::zeros(height, width, CV_8UC1);

    unsigned char *d_input = nullptr, *d_blurred = nullptr, *d_output = nullptr;
    unsigned char *h_pinned_in = nullptr, *h_pinned_out = nullptr;

    cudaMallocHost((void**)&h_pinned_in, img_bytes);
    cudaMallocHost((void**)&h_pinned_out, img_bytes);
    std::memcpy(h_pinned_in, input.data, img_bytes);

    cudaMalloc((void**)&d_input, img_bytes);
    cudaMalloc((void**)&d_blurred, img_bytes);
    cudaMalloc((void**)&d_output, img_bytes);
    cudaMemsetAsync(d_blurred, 0, img_bytes, 0);
    cudaMemsetAsync(d_output, 0, img_bytes, 0);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(d_input, h_pinned_in, img_bytes, cudaMemcpyHostToDevice, stream);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    gaussian_blur_kernel<<<gridSize, blockSize, 0, stream>>>(d_input, d_blurred, width, height);
    sobel_kernel<<<gridSize, blockSize, 0, stream>>>(d_blurred, d_output, width, height);

    cudaMemcpyAsync(h_pinned_out, d_output, img_bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::memcpy(output.data, h_pinned_out, img_bytes);

    cudaFreeHost(h_pinned_in);
    cudaFreeHost(h_pinned_out);
    cudaFree(d_input);
    cudaFree(d_blurred);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
}