#include "gpu_cuda.h"
#include "sar_preprocessing.h"

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <cstring>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

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

__global__ void lee_filter_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    constexpr int radius = 2;
    constexpr float sample_count = 25.0f;
    constexpr float equivalent_looks = 4.0f;
    constexpr float epsilon = 1e-12f;
    float sum = 0.0f;
    float sum_squares = 0.0f;

    for (int ky = -radius; ky <= radius; ++ky) {
        const int sample_y = max(0, min(height - 1, y + ky));
        for (int kx = -radius; kx <= radius; ++kx) {
            const int sample_x = max(0, min(width - 1, x + kx));
            const float value = input[sample_y * width + sample_x];
            sum += value;
            sum_squares += value * value;
        }
    }

    const float local_mean = sum / sample_count;
    const float local_variance =
        fmaxf(0.0f, sum_squares / sample_count - local_mean * local_mean);
    const float noise_variance =
        local_mean * local_mean / equivalent_looks;
    const float weight = fmaxf(
        0.0f,
        (local_variance - noise_variance) / (local_variance + epsilon));
    const float center = input[y * width + x];
    output[y * width + x] = local_mean + weight * (center - local_mean);
}

__global__ void water_segmentation_kernel(
    const float* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height,
    float threshold)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        output[y * width + x] =
            input[y * width + x] <= threshold ? 255 : 0;
    }
}

__global__ void flood_change_kernel(
    const unsigned char* __restrict__ pre_water,
    const unsigned char* __restrict__ post_water,
    unsigned char* __restrict__ new_flood,
    unsigned char* __restrict__ semantic,
    int width,
    int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        const int index = y * width + x;
        const bool was_water = pre_water[index] == 255;
        const bool is_water = post_water[index] == 255;
        new_flood[index] = (!was_water && is_water) ? 255 : 0;
        semantic[index] = !was_water
            ? (is_water ? 2 : 0)
            : (is_water ? 1 : 3);
    }
}

void check_cuda_result(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + " failed: " + cudaGetErrorString(result));
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

void process_lee_cuda(const cv::Mat& input, cv::Mat& output) {
    if (input.empty() || input.type() != CV_32FC1) {
        throw std::invalid_argument(
            "process_lee_cuda requires a non-empty CV_32FC1 image");
    }

    const int width = input.cols;
    const int height = input.rows;
    const size_t image_bytes =
        static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(float);
    cv::Mat contiguous_input = input.isContinuous() ? input : input.clone();
    output.create(input.size(), CV_32FC1);

    float* device_input = nullptr;
    float* device_output = nullptr;
    try {
        check_cuda_result(
            cudaMalloc(reinterpret_cast<void**>(&device_input), image_bytes),
            "cudaMalloc input");
        check_cuda_result(
            cudaMalloc(reinterpret_cast<void**>(&device_output), image_bytes),
            "cudaMalloc output");
        check_cuda_result(
            cudaMemcpy(device_input, contiguous_input.ptr<float>(),
                image_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy input");

        const dim3 block_size(16, 16);
        const dim3 grid_size(
            (width + block_size.x - 1) / block_size.x,
            (height + block_size.y - 1) / block_size.y);
        lee_filter_kernel<<<grid_size, block_size>>>(
            device_input, device_output, width, height);
        check_cuda_result(cudaGetLastError(), "Lee filter kernel launch");
        check_cuda_result(cudaDeviceSynchronize(), "Lee filter kernel execution");
        check_cuda_result(
            cudaMemcpy(output.ptr<float>(), device_output,
                image_bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy output");
    } catch (...) {
        if (device_input) {
            cudaFree(device_input);
        }
        if (device_output) {
            cudaFree(device_output);
        }
        throw;
    }

    check_cuda_result(cudaFree(device_input), "cudaFree input");
    check_cuda_result(cudaFree(device_output), "cudaFree output");
}

void segment_water_cuda(
    const cv::Mat& despeckled,
    cv::Mat& water_mask,
    float threshold) {
    if (despeckled.empty() || despeckled.type() != CV_32FC1) {
        throw std::invalid_argument(
            "segment_water_cuda requires a non-empty CV_32FC1 image");
    }
    if (!std::isfinite(threshold) || threshold < 0.0f || threshold > 1.0f) {
        throw std::invalid_argument("Water threshold must be between 0 and 1");
    }

    const int width = despeckled.cols;
    const int height = despeckled.rows;
    const size_t pixel_count =
        static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t input_bytes = pixel_count * sizeof(float);
    const size_t output_bytes = pixel_count * sizeof(unsigned char);
    cv::Mat contiguous_input =
        despeckled.isContinuous() ? despeckled : despeckled.clone();
    water_mask.create(despeckled.size(), CV_8UC1);

    float* device_input = nullptr;
    unsigned char* device_output = nullptr;
    try {
        check_cuda_result(
            cudaMalloc(reinterpret_cast<void**>(&device_input), input_bytes),
            "cudaMalloc segmentation input");
        check_cuda_result(
            cudaMalloc(reinterpret_cast<void**>(&device_output), output_bytes),
            "cudaMalloc segmentation output");
        check_cuda_result(
            cudaMemcpy(device_input, contiguous_input.ptr<float>(),
                input_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy segmentation input");

        const dim3 block_size(16, 16);
        const dim3 grid_size(
            (width + block_size.x - 1) / block_size.x,
            (height + block_size.y - 1) / block_size.y);
        water_segmentation_kernel<<<grid_size, block_size>>>(
            device_input, device_output, width, height, threshold);
        check_cuda_result(cudaGetLastError(), "water segmentation kernel launch");
        check_cuda_result(
            cudaDeviceSynchronize(), "water segmentation kernel execution");
        check_cuda_result(
            cudaMemcpy(water_mask.ptr<unsigned char>(), device_output,
                output_bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy segmentation output");
    } catch (...) {
        if (device_input) {
            cudaFree(device_input);
        }
        if (device_output) {
            cudaFree(device_output);
        }
        throw;
    }

    check_cuda_result(cudaFree(device_input), "cudaFree segmentation input");
    check_cuda_result(cudaFree(device_output), "cudaFree segmentation output");
}

void detect_flood_change_cuda(
    const cv::Mat& pre_water_mask,
    const cv::Mat& post_water_mask,
    cv::Mat& new_flood_mask,
    cv::Mat& semantic_map) {
    if (!is_binary_water_mask(pre_water_mask) ||
        !is_binary_water_mask(post_water_mask) ||
        pre_water_mask.size() != post_water_mask.size()) {
        throw std::invalid_argument(
            "Flood change detection requires matching binary CV_8UC1 water masks");
    }

    const int width = pre_water_mask.cols;
    const int height = pre_water_mask.rows;
    const size_t image_bytes =
        static_cast<size_t>(width) * static_cast<size_t>(height);
    cv::Mat contiguous_pre =
        pre_water_mask.isContinuous() ? pre_water_mask : pre_water_mask.clone();
    cv::Mat contiguous_post =
        post_water_mask.isContinuous() ? post_water_mask : post_water_mask.clone();
    new_flood_mask.create(pre_water_mask.size(), CV_8UC1);
    semantic_map.create(pre_water_mask.size(), CV_8UC1);

    unsigned char* device_pre = nullptr;
    unsigned char* device_post = nullptr;
    unsigned char* device_flood = nullptr;
    unsigned char* device_semantic = nullptr;
    try {
        check_cuda_result(
            cudaMalloc(reinterpret_cast<void**>(&device_pre), image_bytes),
            "cudaMalloc pre-water mask");
        check_cuda_result(
            cudaMalloc(reinterpret_cast<void**>(&device_post), image_bytes),
            "cudaMalloc post-water mask");
        check_cuda_result(
            cudaMalloc(reinterpret_cast<void**>(&device_flood), image_bytes),
            "cudaMalloc new-flood mask");
        check_cuda_result(
            cudaMalloc(reinterpret_cast<void**>(&device_semantic), image_bytes),
            "cudaMalloc semantic map");
        check_cuda_result(
            cudaMemcpy(device_pre, contiguous_pre.ptr<unsigned char>(),
                image_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy pre-water mask");
        check_cuda_result(
            cudaMemcpy(device_post, contiguous_post.ptr<unsigned char>(),
                image_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy post-water mask");

        const dim3 block_size(16, 16);
        const dim3 grid_size(
            (width + block_size.x - 1) / block_size.x,
            (height + block_size.y - 1) / block_size.y);
        flood_change_kernel<<<grid_size, block_size>>>(
            device_pre, device_post, device_flood, device_semantic, width, height);
        check_cuda_result(cudaGetLastError(), "flood-change kernel launch");
        check_cuda_result(cudaDeviceSynchronize(), "flood-change kernel execution");
        check_cuda_result(
            cudaMemcpy(new_flood_mask.ptr<unsigned char>(), device_flood,
                image_bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy new-flood mask");
        check_cuda_result(
            cudaMemcpy(semantic_map.ptr<unsigned char>(), device_semantic,
                image_bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy semantic map");
    } catch (...) {
        if (device_pre) cudaFree(device_pre);
        if (device_post) cudaFree(device_post);
        if (device_flood) cudaFree(device_flood);
        if (device_semantic) cudaFree(device_semantic);
        throw;
    }

    check_cuda_result(cudaFree(device_pre), "cudaFree pre-water mask");
    check_cuda_result(cudaFree(device_post), "cudaFree post-water mask");
    check_cuda_result(cudaFree(device_flood), "cudaFree new-flood mask");
    check_cuda_result(cudaFree(device_semantic), "cudaFree semantic map");
}

void cleanup_flood_cuda(
    cv::Mat& flood_mask,
    cv::Mat& semantic_map,
    int minimum_patch_pixels) {
    cleanup_flood_classification(
        flood_mask, semantic_map, minimum_patch_pixels);
}