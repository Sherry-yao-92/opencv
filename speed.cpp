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

    if (perimeter_original > 0) {
        results.circularity_original = (2 * sqrt(M_PI * results.area_original)) / perimeter_original;
    } else {
        results.circularity_original = 0;
    }

    convexHull(cnt, results.hull);

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

    results.contour = std::move(cnt);

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

    // 使用固定大小的內核
    static const Mat kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));

    Mat morph;
    morphologyEx(binary, morph, MORPH_CLOSE, kernel);

    vector<vector<Point>> contours;
    findContours(morph, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 找到最大的輪廓
    auto max_contour = max_element(contours.begin(), contours.end(),
        [](const vector<Point>& c1, const vector<Point>& c2) {
            return contourArea(c1) < contourArea(c2);
        });

    if (max_contour != contours.end()) {
        // 只處理最大的輪廓
        vector<vector<Point>> single_contour = {*max_contour};
        return calculate_contour_metrics(single_contour);
    }

    return ContourMetrics(); // 如果沒有找到輪廓，返回空的結果
}

void process_file(const string& img_path, const Mat& background, vector<double>& processing_times, vector<ContourMetrics>& results_list) {
    auto start_time = high_resolution_clock::now();
    ContourMetrics results = process_image(img_path, background);
    auto end_time = high_resolution_clock::now();

    auto process_time = duration_cast<microseconds>(end_time - start_time).count() / 1e6;
    processing_times.push_back(process_time);
    results_list.push_back(results);
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

    // 預先對背景進行高斯模糊
    GaussianBlur(background, background, Size(3, 3), 0);

    vector<double> processing_times;
    vector<ContourMetrics> results_list;
    vector<string> img_paths;

    for (const auto& entry : fs::directory_iterator(img_folder)) {
        if (entry.path().extension() == ".tiff" && entry.path().filename() != "background.tiff") {
            img_paths.push_back(entry.path().string());
        }
    }

    // 處理所有圖片並記錄時間
    for (const auto& img_path : img_paths) {
        process_file(img_path, background, processing_times, results_list);
    }

    // 計算並顯示平均處理時間
    if (!processing_times.empty()) {
        double total_time = std::accumulate(processing_times.begin(), processing_times.end(), 0.0);
        double average_time = total_time / processing_times.size();
        cout << "Average processing time: " << fixed << setprecision(6) << average_time << " seconds" << endl;
    }

    // 顯示所有圖片和數據
    for (size_t i = 0; i < img_paths.size(); ++i) {
        const auto& img_path = img_paths[i];
        const auto& results = results_list[i];
        double process_time = processing_times[i];

        cout << "Results for " << img_path << ":" << endl;
        cout << "Processing time: " << fixed << setprecision(6) << process_time << " seconds" << endl;
        cout << "Original area: " << results.area_original << endl;
        cout << "Convex Hull area: " << results.area_hull << endl;
        cout << "Area ratio (hull/original): " << results.area_ratio << endl;
        cout << "Original circularity: " << results.circularity_original << endl;
        cout << "Convex Hull circularity: " << results.circularity_hull << endl;
        cout << "Circularity ratio (hull/original): " << results.circularity_ratio << endl;
        cout << endl;

        if (!results.contour.empty()) {
            Mat original_contour_image = Mat::zeros(background.size(), CV_8U);
            Mat hull_contour_image = Mat::zeros(background.size(), CV_8U);

            drawContours(original_contour_image, vector<vector<Point>>{results.contour}, -1, Scalar(255), 1);
            drawContours(hull_contour_image, vector<vector<Point>>{results.hull}, -1, Scalar(255), 1);

            imshow("Original Contour - " + img_path, original_contour_image);
            imshow("Convex Hull - " + img_path, hull_contour_image);
            waitKey(0);
            destroyAllWindows();
        } else {
            cout << "No contours found for " << img_path << endl;
        }
    }

    return 0;
}
