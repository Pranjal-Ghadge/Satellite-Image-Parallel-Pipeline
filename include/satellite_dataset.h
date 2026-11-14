#ifndef SATELLITE_DATASET_H
#define SATELLITE_DATASET_H

#include <array>
#include <optional>
#include <string>

#include <opencv2/core.hpp>

struct SatelliteRaster {
    std::string source_path;
    cv::Mat pixels;
    cv::Mat valid_mask;
    std::array<double, 6> geotransform{};
    std::string projection_wkt;
    std::string band_data_type;
    std::string unit;
    int band_index = 1;
    std::optional<double> no_data_value;
    std::optional<double> scale;
    std::optional<double> offset;
};

struct SatelliteImagePair {
    SatelliteRaster pre_flood;
    SatelliteRaster post_flood;
};

// Loads one-based bands into Float64 physical values using GDAL's scale/offset.
// valid_mask is 255 for usable pixels and 0 for NoData, masked, or non-finite samples.
// Source NoData, scale, and offset metadata are retained. The pair must already
// share a pixel grid; no reprojection or resampling is performed.
SatelliteImagePair load_aligned_sentinel1_pair(
    const std::string& pre_flood_path,
    const std::string& post_flood_path,
    int band_index = 1);

#endif
