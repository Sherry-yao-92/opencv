#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>
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
};

ContourMetrics calculate_contour_metrics(const vector<Point>& contour) {
    ContourMetrics results;
    
    results.area_original = contourArea(contour);
    double perimeter_original = arcLength(contour, true);

    if (perimeter_original > 0) {
        results.circularity_original = (2 * sqrt(M_PI * results.area_original)) / perimeter_original;
    } else {
        results.circularity_original = 0;
    }

    convexHull(contour, results.hull);

    results.area_hull = contourArea(results.hull);
    double perimeter_hull = arcLength(results.hull, true);

    if (perimeter_hull > 0) {
        results.circularity_hull = (2 * sqrt(M_PI * results.area_hull)) / perimeter_hull;
    } else {
        results.circularity_hull = 0;
    }

    if (results.area_original > 0) {
        results.area_ratio = results.area_hull / results.area_original;
    } else {
        results.area_ratio = 0;
    }

    if (results.circularity_original > 0) {
        results.circularity_ratio = results.circularity_hull / results.circularity_original;
    } else {
        results.circularity_ratio = 0;
    }

    results.contour = contour;

    return results;
}

ContourMetrics process_image(const string& img_path, const Mat& background) {
    Mat img = imread(img_path, IMREAD_GRAYSCALE);
    if (img.empty()) {
        cout << "Error: Unable to read image: " << img_path << endl;
        return ContourMetrics();
    }

    Mat blur_img, subtract_result, binary;

    GaussianBlur(img, blur_img, Size(3, 3), 0);
    subtract(background, blur_img, subtract_result);
    threshold(subtract_result, binary, 10, 255, THRESH_BINARY);

    static const Mat kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));

    Mat morph;
    morphologyEx(binary, morph, MORPH_CLOSE, kernel);

    vector<vector<Point>> contours;
    findContours(morph, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (!contours.empty()) {
        auto max_contour = *max_element(contours.begin(), contours.end(),
            [](const vector<Point>& c1, const vector<Point>& c2) {
                return contourArea(c1) < contourArea(c2);
            });
        return calculate_contour_metrics(max_contour);
    }

    return ContourMetrics();
}

vector<ContourMetrics> process_images_sequential(const vector<string>& img_paths, const Mat& background) {
    vector<ContourMetrics> results;
    results.reserve(img_paths.size());
    for (const auto& img_path : img_paths) {
        results.push_back(process_image(img_path, background));
    }
    return results;
}

vector<ContourMetrics> process_images_parallel(const vector<string>& img_paths, const Mat& background) {
    vector<ContourMetrics> results(img_paths.size());
    parallel_for_(Range(0, static_cast<int>(img_paths.size())), [&](const Range& range) {
        for (int i = range.start; i < range.end; i++) {
            results[i] = process_image(img_paths[i], background);
        }
    });
    return results;
}

void display_results(const vector<string>& img_paths, const vector<ContourMetrics>& results, const Mat& background, const string& execution_type) {
    cout << execution_type << " execution results:" << endl;
    for (size_t i = 0; i < img_paths.size(); ++i) {
        const auto& img_path = img_paths[i];
        const auto& result = results[i];

        cout << "Results for " << img_path << ":" << endl;
        cout << "Original area: " << result.area_original << endl;
        cout << "Convex Hull area: " << result.area_hull << endl;
        cout << "Area ratio (hull/original): " << result.area_ratio << endl;
        cout << "Original circularity: " << result.circularity_original << endl;
        cout << "Convex Hull circularity: " << result.circularity_hull << endl;
        cout << "Circularity ratio (hull/original): " << result.circularity_ratio << endl;
        cout << endl;

        if (!result.contour.empty()) {
            Mat original_contour_image = Mat::zeros(background.size(), CV_8U);
            Mat hull_contour_image = Mat::zeros(background.size(), CV_8U);

            drawContours(original_contour_image, vector<vector<Point>>{result.contour}, -1, Scalar(255), 1);
            drawContours(hull_contour_image, vector<vector<Point>>{result.hull}, -1, Scalar(255), 1);

            imshow(execution_type + " - Original Contour - " + img_path, original_contour_image);
            imshow(execution_type + " - Convex Hull - " + img_path, hull_contour_image);
            waitKey(0);
            destroyAllWindows();
        } else {
            cout << "No contours found for " << img_path << endl;
        }
    }
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

    GaussianBlur(background, background, Size(3, 3), 0);

    vector<string> img_paths;
    for (const auto& entry : fs::directory_iterator(img_folder)) {
        if (entry.path().extension() == ".tiff" && entry.path().filename() != "background.tiff") {
            img_paths.push_back(entry.path().string());
        }
    }

    // 測試順序執行
    auto start_sequential = high_resolution_clock::now();
    auto results_sequential = process_images_sequential(img_paths, background);
    auto end_sequential = high_resolution_clock::now();
    auto time_sequential = duration_cast<microseconds>(end_sequential - start_sequential).count() / 1e6;

    // 測試並行執行
    auto start_parallel = high_resolution_clock::now();
    auto results_parallel = process_images_parallel(img_paths, background);
    auto end_parallel = high_resolution_clock::now();
    auto time_parallel = duration_cast<microseconds>(end_parallel - start_parallel).count() / 1e6;

    // 輸出結果
    cout << fixed << setprecision(6);
    cout << "Sequential execution time: " << time_sequential << " seconds" << endl;
    cout << "Parallel execution time: " << time_parallel << " seconds" << endl;
    cout << "Speed-up: " << time_sequential / time_parallel << "x" << endl;

    // 顯示順序執行的結果
    display_results(img_paths, results_sequential, background, "Sequential");

    // 顯示並行執行的結果
    display_results(img_paths, results_parallel, background, "Parallel");

    return 0;
}
