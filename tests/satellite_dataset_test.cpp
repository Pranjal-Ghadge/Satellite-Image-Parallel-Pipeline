#include "satellite_dataset.h"

#include <array>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cpl_conv.h>
#include <gdal_priv.h>
#include <ogr_spatialref.h>

namespace {

struct RasterSpec {
    int width = 3;
    int height = 2;
    std::array<double, 6> transform{500000.0, 10.0, 0.0, 4600000.0, 0.0, -10.0};
    int epsg = 32632;
    bool include_crs = true;
    int bands = 1;
    GDALDataType data_type = GDT_Float32;
    bool include_mask_band = false;
};

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        path_ = std::filesystem::temp_directory_path() /
            ("aakashnetra-gdal-tests-" +
             std::to_string(std::filesystem::file_time_type::clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(path_);
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    std::filesystem::path file(const std::string& name) const {
        return path_ / name;
    }

private:
    std::filesystem::path path_;
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string projection_wkt(int epsg) {
    OGRSpatialReference reference;
    require(reference.importFromEPSG(epsg) == OGRERR_NONE, "Unable to create test CRS.");
    reference.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    char* wkt = nullptr;
    require(reference.exportToWkt(&wkt) == OGRERR_NONE, "Unable to export test CRS.");
    const std::string result(wkt);
    CPLFree(wkt);
    return result;
}

bool crs_matches_epsg(const std::string& wkt, int epsg) {
    OGRSpatialReference actual;
    OGRSpatialReference expected;
    if (actual.SetFromUserInput(wkt.c_str()) != OGRERR_NONE ||
        expected.importFromEPSG(epsg) != OGRERR_NONE) {
        return false;
    }
    actual.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    expected.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    return actual.IsSame(&expected) != 0;
}

void create_geotiff(const std::filesystem::path& path, const RasterSpec& spec) {
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    require(driver != nullptr, "GDAL GeoTIFF driver is unavailable.");

    GDALDataset* dataset = driver->Create(
        path.string().c_str(), spec.width, spec.height, spec.bands, spec.data_type, nullptr);
    require(dataset != nullptr, "Unable to create test GeoTIFF.");

    if (dataset->SetGeoTransform(const_cast<double*>(spec.transform.data())) != CE_None) {
        GDALClose(dataset);
        throw std::runtime_error("Unable to set test geotransform.");
    }
    if (spec.include_crs) {
        const std::string wkt = projection_wkt(spec.epsg);
        if (dataset->SetProjection(wkt.c_str()) != CE_None) {
            GDALClose(dataset);
            throw std::runtime_error("Unable to set test CRS.");
        }
    }

    for (int band_index = 1; band_index <= spec.bands; ++band_index) {
        GDALRasterBand* band = dataset->GetRasterBand(band_index);
        if (GDALDataTypeIsComplex(spec.data_type)) {
            std::vector<float> samples(
                static_cast<std::size_t>(spec.width * spec.height * 2),
                static_cast<float>(band_index));
            if (band->RasterIO(
                    GF_Write, 0, 0, spec.width, spec.height, samples.data(),
                    spec.width, spec.height, spec.data_type, 0, 0, nullptr) != CE_None) {
                GDALClose(dataset);
                throw std::runtime_error("Unable to populate complex test GeoTIFF.");
            }
        } else {
            std::vector<float> samples(
                static_cast<std::size_t>(spec.width * spec.height),
                static_cast<float>(band_index));
            if (band_index == 1) {
                samples[1] = -9999.0F;
            }
            if (band->RasterIO(
                    GF_Write, 0, 0, spec.width, spec.height, samples.data(),
                    spec.width, spec.height, GDT_Float32, 0, 0, nullptr) != CE_None) {
                GDALClose(dataset);
                throw std::runtime_error("Unable to populate test GeoTIFF.");
            }
        }
        if (band_index == 1 && !GDALDataTypeIsComplex(spec.data_type)) {
            band->SetNoDataValue(-9999.0);
            band->SetScale(0.01);
            band->SetOffset(-2.0);
            band->SetUnitType("amplitude");
            if (spec.include_mask_band) {
                if (band->CreateMaskBand(GMF_PER_DATASET) != CE_None) {
                    GDALClose(dataset);
                    throw std::runtime_error("Unable to create test validity mask.");
                }
                GDALRasterBand* mask_band = band->GetMaskBand();
                std::vector<unsigned char> mask(
                    static_cast<std::size_t>(spec.width * spec.height), 255);
                mask.back() = 0;
                if (mask_band == nullptr ||
                    mask_band->RasterIO(
                        GF_Write, 0, 0, spec.width, spec.height, mask.data(),
                        spec.width, spec.height, GDT_Byte, 0, 0, nullptr) != CE_None) {
                    GDALClose(dataset);
                    throw std::runtime_error("Unable to populate test validity mask.");
                }
            }
        }
    }
    GDALClose(dataset);
}

void expect_failure(
    const std::function<void()>& operation,
    const std::string& expected_message_fragment) {
    try {
        operation();
    } catch (const std::exception& error) {
        require(
            std::string(error.what()).find(expected_message_fragment) != std::string::npos,
            "Unexpected error message: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("Expected failure containing: " + expected_message_fragment);
}

void test_loads_scaled_samples_and_preserves_metadata(const TemporaryDirectory& directory) {
    const auto pre_path = directory.file("pre.tif");
    const auto post_path = directory.file("post.tif");
    create_geotiff(pre_path, RasterSpec{});
    create_geotiff(post_path, RasterSpec{});

    const SatelliteImagePair pair =
        load_aligned_sentinel1_pair(pre_path.string(), post_path.string());
    require(pair.pre_flood.pixels.type() == CV_64FC1, "Samples should be loaded as Float64.");
    require(pair.pre_flood.pixels.cols == 3 && pair.pre_flood.pixels.rows == 2,
            "Raster dimensions were not preserved.");
    require(std::abs(pair.pre_flood.pixels.at<double>(0, 0) - (-1.99)) < 1e-12,
            "GDAL scale and offset were not applied to valid samples.");
    require(pair.pre_flood.valid_mask.type() == CV_8UC1 &&
                pair.pre_flood.valid_mask.at<uchar>(0, 0) == 255,
            "Valid sample was not marked as usable.");
    require(pair.pre_flood.valid_mask.at<uchar>(0, 1) == 0 &&
                pair.pre_flood.pixels.at<double>(0, 1) == 0.0,
            "NoData sample was not excluded from subsequent processing.");
    require(crs_matches_epsg(pair.pre_flood.projection_wkt, 32632),
            "CRS metadata was not retained.");
    require(pair.pre_flood.geotransform[0] == 500000.0 &&
                pair.pre_flood.geotransform[1] == 10.0 &&
                pair.pre_flood.geotransform[5] == -10.0,
            "Geotransform metadata was not retained.");
    require(pair.pre_flood.no_data_value && *pair.pre_flood.no_data_value == -9999.0,
            "NoData metadata was not retained.");
    require(pair.pre_flood.scale && *pair.pre_flood.scale == 0.01,
            "Band scale metadata was not retained.");
    require(pair.pre_flood.offset && *pair.pre_flood.offset == -2.0,
            "Band offset metadata was not retained.");
    require(pair.pre_flood.unit == "amplitude", "Band unit metadata was not retained.");
}

void test_invalid_samples_are_excluded(const TemporaryDirectory& directory) {
    RasterSpec spec;
    const auto path = directory.file("invalid-samples.tif");
    create_geotiff(path, spec);

    GDALDataset* dataset = static_cast<GDALDataset*>(
        GDALOpen(path.string().c_str(), GA_Update));
    require(dataset != nullptr, "Unable to reopen test GeoTIFF.");
    GDALRasterBand* band = dataset->GetRasterBand(1);
    std::vector<float> samples{1.0F, -9999.0F, std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::infinity(), 5.0F, 7.0F};
    const CPLErr write_result = band->RasterIO(
        GF_Write, 0, 0, spec.width, spec.height, samples.data(),
        spec.width, spec.height, GDT_Float32, 0, 0, nullptr);
    GDALClose(dataset);
    require(write_result == CE_None, "Unable to write invalid test samples.");

    const SatelliteImagePair pair =
        load_aligned_sentinel1_pair(path.string(), path.string());
    require(pair.pre_flood.valid_mask.at<uchar>(0, 1) == 0,
            "NoData pixel should be invalid.");
    require(pair.pre_flood.valid_mask.at<uchar>(1, 0) == 0,
            "NaN pixel should be invalid.");
    require(pair.pre_flood.valid_mask.at<uchar>(1, 1) == 0,
            "Infinite pixel should be invalid.");
    require(pair.pre_flood.valid_mask.at<uchar>(1, 2) == 255,
            "Finite non-NoData sample should remain valid.");
}

void test_gdal_mask_band_excludes_invalid_pixels(const TemporaryDirectory& directory) {
    RasterSpec spec;
    spec.include_mask_band = true;
    const auto path = directory.file("mask-band.tif");
    create_geotiff(path, spec);

    const SatelliteImagePair pair =
        load_aligned_sentinel1_pair(path.string(), path.string());
    require(pair.pre_flood.valid_mask.at<uchar>(1, 2) == 0,
            "A pixel excluded by the GDAL mask band should be invalid.");
    require(pair.pre_flood.pixels.at<double>(1, 2) == 0.0,
            "A pixel excluded by the GDAL mask band should not carry a usable value.");
}

void test_rejects_mismatched_dimensions(const TemporaryDirectory& directory) {
    RasterSpec different_size;
    different_size.width = 4;
    create_geotiff(directory.file("pre-size.tif"), RasterSpec{});
    create_geotiff(directory.file("post-size.tif"), different_size);
    expect_failure(
        [&] {
            load_aligned_sentinel1_pair(
                directory.file("pre-size.tif").string(),
                directory.file("post-size.tif").string());
        },
        "dimensions do not match");
}

void test_rejects_mismatched_crs(const TemporaryDirectory& directory) {
    RasterSpec different_crs;
    different_crs.epsg = 32633;
    create_geotiff(directory.file("pre-crs.tif"), RasterSpec{});
    create_geotiff(directory.file("post-crs.tif"), different_crs);
    expect_failure(
        [&] {
            load_aligned_sentinel1_pair(
                directory.file("pre-crs.tif").string(),
                directory.file("post-crs.tif").string());
        },
        "coordinate reference systems do not match");
}

void test_rejects_missing_crs(const TemporaryDirectory& directory) {
    RasterSpec missing_crs;
    missing_crs.include_crs = false;
    create_geotiff(directory.file("no-crs.tif"), missing_crs);
    expect_failure(
        [&] {
            load_aligned_sentinel1_pair(
                directory.file("no-crs.tif").string(),
                directory.file("no-crs.tif").string());
        },
        "missing its coordinate reference system");
}

void test_rejects_different_pixel_sizes(const TemporaryDirectory& directory) {
    RasterSpec different_resolution;
    different_resolution.transform[1] = 20.0;
    create_geotiff(directory.file("pre-resolution.tif"), RasterSpec{});
    create_geotiff(directory.file("post-resolution.tif"), different_resolution);
    expect_failure(
        [&] {
            load_aligned_sentinel1_pair(
                directory.file("pre-resolution.tif").string(),
                directory.file("post-resolution.tif").string());
        },
        "pixel sizes or grid orientations do not match");
}

void test_rejects_fractional_pixel_origin_offset(const TemporaryDirectory& directory) {
    RasterSpec half_pixel_shift;
    half_pixel_shift.transform[0] += 5.0;
    create_geotiff(directory.file("pre-origin.tif"), RasterSpec{});
    create_geotiff(directory.file("post-origin.tif"), half_pixel_shift);
    expect_failure(
        [&] {
            load_aligned_sentinel1_pair(
                directory.file("pre-origin.tif").string(),
                directory.file("post-origin.tif").string());
        },
        "origins are not aligned");
}

void test_accepts_matching_rotated_geotransforms(const TemporaryDirectory& directory) {
    RasterSpec rotated;
    rotated.transform = {500000.0, 10.0, 2.0, 4600000.0, 1.0, -10.0};
    const auto pre_path = directory.file("pre-rotated.tif");
    const auto post_path = directory.file("post-rotated.tif");
    create_geotiff(pre_path, rotated);
    create_geotiff(post_path, rotated);

    const SatelliteImagePair pair =
        load_aligned_sentinel1_pair(pre_path.string(), post_path.string());
    require(pair.pre_flood.geotransform[2] == 2.0 &&
                pair.pre_flood.geotransform[4] == 1.0,
            "Rotated geotransform was not retained.");
}

void test_rejects_half_pixel_shift_on_rotated_grid(const TemporaryDirectory& directory) {
    RasterSpec rotated;
    rotated.transform = {500000.0, 10.0, 2.0, 4600000.0, 1.0, -10.0};
    RasterSpec shifted = rotated;
    shifted.transform[0] += 5.0;
    shifted.transform[3] += 0.5;
    const auto pre_path = directory.file("pre-rotated-shift.tif");
    const auto post_path = directory.file("post-rotated-shift.tif");
    create_geotiff(pre_path, rotated);
    create_geotiff(post_path, shifted);
    expect_failure(
        [&] { load_aligned_sentinel1_pair(pre_path.string(), post_path.string()); },
        "origins are not aligned");
}

void test_rejects_degenerate_geotransform(const TemporaryDirectory& directory) {
    RasterSpec degenerate;
    degenerate.transform = {500000.0, 10.0, 20.0, 4600000.0, 5.0, 10.0};
    const auto path = directory.file("degenerate-grid.tif");
    create_geotiff(path, degenerate);
    expect_failure(
        [&] { load_aligned_sentinel1_pair(path.string(), path.string()); },
        "invalid geotransform");
}

void test_reports_missing_files_and_invalid_bands(const TemporaryDirectory& directory) {
    const auto valid_path = directory.file("valid.tif");
    create_geotiff(valid_path, RasterSpec{});
    expect_failure(
        [&] {
            load_aligned_sentinel1_pair(
                directory.file("missing.tif").string(), valid_path.string());
        },
        "Unable to open GeoTIFF");
    expect_failure(
        [&] {
            load_aligned_sentinel1_pair(valid_path.string(), valid_path.string(), 2);
        },
        "does not contain requested band 2");
}

void test_loads_selected_one_based_band(const TemporaryDirectory& directory) {
    RasterSpec multiband;
    multiband.bands = 2;
    const auto pre_path = directory.file("pre-multiband.tif");
    const auto post_path = directory.file("post-multiband.tif");
    create_geotiff(pre_path, multiband);
    create_geotiff(post_path, multiband);

    const SatelliteImagePair pair =
        load_aligned_sentinel1_pair(pre_path.string(), post_path.string(), 2);
    require(pair.pre_flood.band_index == 2, "Requested band index was not retained.");
    require(std::abs(pair.pre_flood.pixels.at<double>(0, 0) - 2.0) < 1e-12,
            "The requested one-based band was not loaded.");
}

void test_rejects_complex_sar_band(const TemporaryDirectory& directory) {
    RasterSpec complex;
    complex.data_type = GDT_CFloat32;
    const auto path = directory.file("complex-sar.tif");
    create_geotiff(path, complex);
    expect_failure(
        [&] { load_aligned_sentinel1_pair(path.string(), path.string()); },
        "complex-valued SAR samples");
}

}  // namespace

int main() {
    GDALAllRegister();
    try {
        TemporaryDirectory directory;
        test_loads_scaled_samples_and_preserves_metadata(directory);
        test_invalid_samples_are_excluded(directory);
        test_gdal_mask_band_excludes_invalid_pixels(directory);
        test_rejects_mismatched_dimensions(directory);
        test_rejects_mismatched_crs(directory);
        test_rejects_missing_crs(directory);
        test_rejects_different_pixel_sizes(directory);
        test_rejects_fractional_pixel_origin_offset(directory);
        test_accepts_matching_rotated_geotransforms(directory);
        test_rejects_half_pixel_shift_on_rotated_grid(directory);
        test_rejects_degenerate_geotransform(directory);
        test_reports_missing_files_and_invalid_bands(directory);
        test_loads_selected_one_based_band(directory);
        test_rejects_complex_sar_band(directory);
        std::cout << "All satellite dataset tests passed.\n";
    } catch (const std::exception& error) {
        std::cerr << "Satellite dataset test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
