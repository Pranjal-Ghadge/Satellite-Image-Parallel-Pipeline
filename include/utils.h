#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <chrono>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double stop_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
};

void log_performance(const std::string& mode, int width, int height, double execution_time_ms, double baseline_time_ms = 0.0);

#endif