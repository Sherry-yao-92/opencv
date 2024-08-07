#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <iomanip>
#include <tbb/parallel_for.h>
#include <tbb/concurrent_vector.h>
#include <filesystem>
#include <algorithm>
#include <map>
#include <numeric>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace cv;
using namespace std;
using namespace std::chrono;
namespace fs = std::filesystem;

struct ContourMetrics {
    double area_original;
    double area_hull;
    double area_ratio;
    double circularity_original;
    double circularity_hull;
    double circularity_ratio;
    vector<Point> contour;
    vector<Point> hull;
    long long process_time;
};

ContourMetrics calculate_contour_metrics(const vector<vector<Point>>& contours) {
    ContourMetrics results;
    if (contours.empty()) {
        return results;
    }

    auto cnt = *max_element(contours.begin(), contours.end(),
        [](const vector<Point>& c1, const vector<Point>& c2) {
            return contourArea(c1) < contourArea(c2);
        });

    results.area_original = contourArea(cnt);
    double perimeter_original = arcLength(cnt, true);
    results.circularity_original = (2 * sqrt(M_PI * results.area_original)) / perimeter_original;

    convexHull(cnt, results.hull);

    results.area_hull = contourArea(results.hull);
    double perimeter_hull = arcLength(results.hull, true);
    results.circularity_hull = (2 * sqrt(M_PI * results.area_hull)) / perimeter_hull;

    results.area_ratio = results.area_hull / results.area_original;
    results.circularity_ratio = results.circularity_hull / results.circularity_original;

    results.contour = cnt;

    return results;
}

ContourMetrics process_image(const Mat& img, const Mat& background) {
    Mat blur_img, blur_background, substract, binary, erode1, dilate1, dilate2, erode2, edge;

    GaussianBlur(img, blur_img, Size(3, 3), 0);
    GaussianBlur(background, blur_background, Size(3, 3), 0);

    subtract(blur_background, blur_img, substract);
    threshold(substract, binary, 10, 255, THRESH_BINARY);

    Mat kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));

    erode(binary, erode1, kernel);
    dilate(erode1, dilate1, kernel);
    dilate(dilate1, dilate2, kernel);
    erode(dilate2, erode2, kernel);

    Canny(erode2, edge, 50, 150);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(edge, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    return calculate_contour_metrics(contours);
}

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    cout << "OpenCV version: " << CV_VERSION << endl;

    string img_folder = "Test_images\\Slight under focus\\";
    string background_path = img_folder + "background.tiff";

    Mat background = imread(background_path, IMREAD_GRAYSCALE);
    if (background.empty()) {
        cout << "Error: Unable to read background image: " << background_path << endl;
        return -1;
    }

    vector<fs::path> image_paths;
    for (const auto & entry : fs::directory_iterator(img_folder)) {
        if (entry.path().extension() == ".tiff" && entry.path().filename() != "background.tiff") {
            image_paths.push_back(entry.path());
        }
    }

    // 使用TBB并行排序
    sort(image_paths.begin(), image_paths.end(), [](const fs::path& a, const fs::path& b) {
        return a.filename() < b.filename();
    });

    tbb::concurrent_vector<long long> processing_times;
    tbb::concurrent_vector<ContourMetrics> results;

    auto start_time = high_resolution_clock::now();

    tbb::parallel_for(tbb::blocked_range<size_t>(0, image_paths.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                Mat img = imread(image_paths[i].string(), IMREAD_GRAYSCALE);
                if (img.empty()) {
                    cout << "Error: Unable to read image: " << image_paths[i] << endl;
                    continue;
                }

                auto img_start_time = high_resolution_clock::now();
                ContourMetrics result = process_image(img, background);
                auto img_end_time = high_resolution_clock::now();

                long long process_time = duration_cast<microseconds>(img_end_time - img_start_time).count();
                processing_times.push_back(process_time);
                results.push_back(result);
            }
        });

    auto end_time = high_resolution_clock::now();
    long long total_time = duration_cast<microseconds>(end_time - start_time).count();

    cout << "Total processing time: " << total_time << " microseconds" << endl;
    cout << "Total images processed: " << processing_times.size() << endl;
    if (!processing_times.empty()) {
        long long total_processing_time = std::accumulate(processing_times.begin(), processing_times.end(), 0LL);
        double average_time = static_cast<double>(total_processing_time) / processing_times.size();
        cout << "Average processing time: " << fixed << setprecision(2) << average_time << " microseconds per image" << endl;
    }
    cout << endl;

    for (size_t i = 0; i < image_paths.size(); ++i) {
        const auto& metrics = results[i];

        cout << "Processing " << image_paths[i].filename() << ":" << endl;
        cout << fixed << setprecision(6);
        cout << "Processing time: " << processing_times[i] << " microseconds" << endl;
        cout << "Original area: " << metrics.area_original << endl;
        cout << "Convex Hull area: " << metrics.area_hull << endl;
        cout << "Area ratio (hull/original): " << metrics.area_ratio << endl;
        cout << "Original circularity: " << metrics.circularity_original << endl;
        cout << "Convex Hull circularity: " << metrics.circularity_hull << endl;
        cout << "Circularity ratio (hull/original): " << metrics.circularity_ratio << endl;
        cout << endl;

        Mat original_contour_image = Mat::zeros(background.size(), CV_8U);
        Mat hull_contour_image = Mat::zeros(background.size(), CV_8U);

        if (!metrics.contour.empty()) {
            drawContours(original_contour_image, vector<vector<Point>>{metrics.contour}, -1, Scalar(255), 1);
            drawContours(hull_contour_image, vector<vector<Point>>{metrics.hull}, -1, Scalar(255), 1);

            imshow("Original Contour - " + image_paths[i].filename().string(), original_contour_image);
            imshow("Convex Hull - " + image_paths[i].filename().string(), hull_contour_image);
            waitKey(0);
            destroyAllWindows();
        } else {
            cout << "No contours found for this image." << endl;
        }
    }

    return 0;
}