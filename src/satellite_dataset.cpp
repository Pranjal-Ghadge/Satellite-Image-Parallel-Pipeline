#include "satellite_dataset.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include <cpl_error.h>
#include <gdal_priv.h>
#include <ogr_spatialref.h>

namespace {

constexpr double kTransformRelativeTolerance = 1e-9;
constexpr double kOriginPixelTolerance = 1e-6;

double normalized_transform_determinant(
    const std::array<double, 6>& geotransform,
    const std::string& path) {
    const double pixel_scale = std::max({
        std::abs(geotransform[1]), std::abs(geotransform[2]),
        std::abs(geotransform[4]), std::abs(geotransform[5])});
    if (pixel_scale == 0.0) {
        throw std::runtime_error(
            "GeoTIFF '" + path + "' has an invalid geotransform: pixel grid is degenerate.");
    }

    const double determinant =
        (geotransform[1] / pixel_scale) * (geotransform[5] / pixel_scale) -
        (geotransform[2] / pixel_scale) * (geotransform[4] / pixel_scale);
    if (!std::isfinite(determinant) ||
        std::abs(determinant) <= std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error(
            "GeoTIFF '" + path + "' has an invalid geotransform: pixel grid is degenerate.");
    }
    return determinant;
}

struct DatasetCloser {
    void operator()(GDALDataset* dataset) const {
        if (dataset != nullptr) {
            GDALClose(dataset);
        }
    }
};

using DatasetPtr = std::unique_ptr<GDALDataset, DatasetCloser>;

std::string gdal_error_message() {
    const char* message = CPLGetLastErrorMsg();
    return message != nullptr && message[0] != '\0'
        ? message
        : "GDAL did not provide additional error details";
}

SatelliteRaster read_raster_band(const std::string& path, int band_index) {
    CPLErrorReset();
    DatasetPtr dataset(static_cast<GDALDataset*>(
        GDALOpenEx(path.c_str(), GDAL_OF_RASTER | GDAL_OF_READONLY, nullptr, nullptr, nullptr)));
    if (!dataset) {
        throw std::runtime_error(
            "Unable to open GeoTIFF '" + path + "': " + gdal_error_message());
    }

    const GDALDriver* driver = dataset->GetDriver();
    if (driver == nullptr || std::string(driver->GetDescription()) != "GTiff") {
        throw std::runtime_error("Input '" + path + "' is not a GeoTIFF dataset.");
    }

    const int width = dataset->GetRasterXSize();
    const int height = dataset->GetRasterYSize();
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("GeoTIFF '" + path + "' has invalid raster dimensions.");
    }
    if (band_index < 1 || band_index > dataset->GetRasterCount()) {
        throw std::runtime_error(
            "GeoTIFF '" + path + "' does not contain requested band " +
            std::to_string(band_index) + ".");
    }

    SatelliteRaster raster;
    raster.source_path = path;
    raster.band_index = band_index;

    CPLErrorReset();
    if (dataset->GetGeoTransform(raster.geotransform.data()) != CE_None) {
        throw std::runtime_error(
            "Unable to read the geotransform from GeoTIFF '" + path + "': " +
            gdal_error_message());
    }
    for (double coefficient : raster.geotransform) {
        if (!std::isfinite(coefficient)) {
            throw std::runtime_error(
                "GeoTIFF '" + path + "' contains a non-finite geotransform coefficient.");
        }
    }

    const char* projection = dataset->GetProjectionRef();
    if (projection == nullptr || projection[0] == '\0') {
        throw std::runtime_error("GeoTIFF '" + path + "' is missing its coordinate reference system.");
    }
    raster.projection_wkt = projection;

    GDALRasterBand* band = dataset->GetRasterBand(band_index);
    if (band == nullptr) {
        throw std::runtime_error(
            "Unable to access band " + std::to_string(band_index) +
            " in GeoTIFF '" + path + "'.");
    }
    const GDALDataType data_type = band->GetRasterDataType();
    if (GDALDataTypeIsComplex(data_type)) {
        throw std::runtime_error(
            "GeoTIFF '" + path + "' band " + std::to_string(band_index) +
            " contains complex-valued SAR samples; provide a real-valued backscatter "
            "GeoTIFF instead.");
    }
    if (data_type == GDT_Unknown) {
        throw std::runtime_error(
            "GeoTIFF '" + path + "' band " + std::to_string(band_index) +
            " has an unsupported unknown sample type.");
    }
    raster.band_data_type = GDALGetDataTypeName(data_type);

    int has_no_data = FALSE;
    const double no_data = band->GetNoDataValue(&has_no_data);
    if (has_no_data) {
        raster.no_data_value = no_data;
    }

    int has_scale = FALSE;
    const double scale = band->GetScale(&has_scale);
    if (has_scale) {
        raster.scale = scale;
    }

    int has_offset = FALSE;
    const double offset = band->GetOffset(&has_offset);
    if (has_offset) {
        raster.offset = offset;
    }

    const double applied_scale = has_scale ? scale : 1.0;
    const double applied_offset = has_offset ? offset : 0.0;
    if (!std::isfinite(applied_scale) || !std::isfinite(applied_offset)) {
        throw std::runtime_error(
            "GeoTIFF '" + path + "' band " + std::to_string(band_index) +
            " has a non-finite GDAL scale or offset.");
    }

    const char* unit = band->GetUnitType();
    if (unit != nullptr) {
        raster.unit = unit;
    }

    raster.pixels.create(height, width, CV_64FC1);
    CPLErrorReset();
    if (band->RasterIO(
            GF_Read,
            0,
            0,
            width,
            height,
            raster.pixels.ptr<double>(),
            width,
            height,
            GDT_Float64,
            static_cast<GSpacing>(sizeof(double)),
            static_cast<GSpacing>(raster.pixels.step[0]),
            nullptr) != CE_None) {
        throw std::runtime_error(
            "Unable to read band " + std::to_string(band_index) +
            " from GeoTIFF '" + path + "': " + gdal_error_message());
    }

    cv::Mat source_mask;
    if ((band->GetMaskFlags() & GMF_ALL_VALID) == 0) {
        GDALRasterBand* mask_band = band->GetMaskBand();
        if (mask_band == nullptr) {
            throw std::runtime_error(
                "Unable to access the validity mask for band " +
                std::to_string(band_index) + " in GeoTIFF '" + path + "'.");
        }
        source_mask.create(height, width, CV_8UC1);
        CPLErrorReset();
        if (mask_band->RasterIO(
                GF_Read,
                0,
                0,
                width,
                height,
                source_mask.ptr<uchar>(),
                width,
                height,
                GDT_Byte,
                static_cast<GSpacing>(sizeof(uchar)),
                static_cast<GSpacing>(source_mask.step[0]),
                nullptr) != CE_None) {
            throw std::runtime_error(
                "Unable to read the validity mask for band " +
                std::to_string(band_index) + " in GeoTIFF '" + path + "': " +
                gdal_error_message());
        }
    }

    raster.valid_mask = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
    for (int row = 0; row < height; ++row) {
        double* samples = raster.pixels.ptr<double>(row);
        uchar* valid = raster.valid_mask.ptr<uchar>(row);
        const uchar* source_valid = source_mask.empty() ? nullptr : source_mask.ptr<uchar>(row);
        for (int column = 0; column < width; ++column) {
            const double raw_sample = samples[column];
            const bool is_no_data = has_no_data &&
                ((std::isnan(no_data) && std::isnan(raw_sample)) || raw_sample == no_data);
            if (!std::isfinite(raw_sample) || is_no_data ||
                (source_valid != nullptr && source_valid[column] == 0)) {
                samples[column] = 0.0;
                continue;
            }

            const double physical_value = raw_sample * applied_scale + applied_offset;
            if (!std::isfinite(physical_value)) {
                samples[column] = 0.0;
                continue;
            }
            samples[column] = physical_value;
            valid[column] = 255;
        }
    }

    return raster;
}

bool same_crs(const SatelliteRaster& first, const SatelliteRaster& second) {
    OGRSpatialReference first_crs;
    OGRSpatialReference second_crs;
    if (first_crs.SetFromUserInput(first.projection_wkt.c_str()) != OGRERR_NONE ||
        second_crs.SetFromUserInput(second.projection_wkt.c_str()) != OGRERR_NONE) {
        throw std::runtime_error("Unable to parse the coordinate reference system in a GeoTIFF.");
    }
    first_crs.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    second_crs.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    return first_crs.IsSame(&second_crs) != 0;
}

void validate_same_grid(const SatelliteRaster& pre_flood, const SatelliteRaster& post_flood) {
    if (pre_flood.pixels.size() != post_flood.pixels.size()) {
        throw std::runtime_error(
            "Pre-flood and post-flood GeoTIFF dimensions do not match (" +
            std::to_string(pre_flood.pixels.cols) + "x" + std::to_string(pre_flood.pixels.rows) +
            " versus " + std::to_string(post_flood.pixels.cols) + "x" +
            std::to_string(post_flood.pixels.rows) + ").");
    }

    if (!same_crs(pre_flood, post_flood)) {
        throw std::runtime_error(
            "Pre-flood and post-flood GeoTIFF coordinate reference systems do not match.");
    }

    const auto& first = pre_flood.geotransform;
    const auto& second = post_flood.geotransform;
    const double pixel_scale = std::max({
        std::abs(first[1]), std::abs(first[2]), std::abs(first[4]), std::abs(first[5]),
        std::abs(second[1]), std::abs(second[2]), std::abs(second[4]), std::abs(second[5])});
    if (pixel_scale == 0.0) {
        throw std::runtime_error("GeoTIFF geotransform has a degenerate pixel grid.");
    }

    for (int coefficient : {1, 2, 4, 5}) {
        const double coefficient_scale =
            std::max(std::abs(first[coefficient]), std::abs(second[coefficient]));
        if (std::abs(first[coefficient] - second[coefficient]) >
            coefficient_scale * kTransformRelativeTolerance) {
            throw std::runtime_error(
                "Pre-flood and post-flood GeoTIFF pixel sizes or grid orientations do not match.");
        }
    }

    const double origin_delta_x = second[0] - first[0];
    const double origin_delta_y = second[3] - first[3];
    const double normalized_determinant =
        normalized_transform_determinant(first, pre_flood.source_path);
    const double column_offset =
        ((origin_delta_x / pixel_scale) * (first[5] / pixel_scale) -
         (origin_delta_y / pixel_scale) * (first[2] / pixel_scale)) /
        normalized_determinant;
    const double row_offset =
        ((origin_delta_y / pixel_scale) * (first[1] / pixel_scale) -
         (origin_delta_x / pixel_scale) * (first[4] / pixel_scale)) /
        normalized_determinant;
    if (!std::isfinite(column_offset) || !std::isfinite(row_offset) ||
        std::abs(column_offset) > kOriginPixelTolerance ||
        std::abs(row_offset) > kOriginPixelTolerance) {
        throw std::runtime_error(
            "Pre-flood and post-flood GeoTIFF origins are not aligned to the same pixel grid.");
    }
}

}  // namespace

SatelliteImagePair load_aligned_sentinel1_pair(
    const std::string& pre_flood_path,
    const std::string& post_flood_path,
    int band_index) {
    if (band_index < 1) {
        throw std::runtime_error("Band index must be a positive, one-based number.");
    }

    GDALAllRegister();
    SatelliteImagePair pair;
    pair.pre_flood = read_raster_band(pre_flood_path, band_index);
    pair.post_flood = read_raster_band(post_flood_path, band_index);
    validate_same_grid(pair.pre_flood, pair.post_flood);
    return pair;
}
