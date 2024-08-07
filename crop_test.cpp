#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>

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

bool is_contour_complete(const vector<Point>& contour, const Size& image_size) {
    for (const Point& point : contour) {
        if (point.x <= 0 || point.y <= 0 || point.x >= image_size.width - 1 || point.y >= image_size.height - 1) {
            return false;
        }
    }
    return true;
}

ContourMetrics process_image(const string& img_path, const Mat& background) {
    Mat img = imread(img_path, IMREAD_GRAYSCALE);
    if (img.empty()) {
        cout << "Error: Unable to read image: " << img_path << endl;
        return ContourMetrics();
    }

    Mat blur_img, blur_background;
    GaussianBlur(img, blur_img, Size(3, 3), 0);
    GaussianBlur(background, blur_background, Size(3, 3), 0);

    Mat substract;
    subtract(blur_background, blur_img, substract);

    Mat binary;
    threshold(substract, binary, 10, 255, THRESH_BINARY);

    Mat kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));

    Mat erode1, dilate1, dilate2, erode2;
    erode(binary, erode1, kernel);
    dilate(erode1, dilate1, kernel);
    dilate(dilate1, dilate2, kernel);
    erode(dilate2, erode2, kernel);

    Mat edge;
    Canny(erode2, edge, 50, 150);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(edge, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    // 如果沒有找到輪廓或找到的輪廓數量超過一個，返回空結果
    if (contours.empty() || contours.size() > 1) {
        return ContourMetrics();
    }

    // 檢查輪廓是否完整
    if (!is_contour_complete(contours[0], img.size())) {
        return ContourMetrics();
    }

    return calculate_contour_metrics(contours);
}

void process_and_compare(const string& original_path, const string& cropped_path, const Mat& original_background, const Mat& cropped_background) {
    auto start_time_original = high_resolution_clock::now();
    ContourMetrics original_results = process_image(original_path, original_background);
    auto end_time_original = high_resolution_clock::now();

    auto start_time_cropped = high_resolution_clock::now();
    ContourMetrics cropped_results = process_image(cropped_path, cropped_background);
    auto end_time_cropped = high_resolution_clock::now();

    // 檢查是否找到輪廓
    if (original_results.contour.empty() || cropped_results.contour.empty()) {
        cout << "Skipping " << fs::path(original_path).filename() << " due to no contours found or incomplete contours." << endl;
        return;
    }

    auto process_time_original = duration_cast<microseconds>(end_time_original - start_time_original).count() / 1e6;
    auto process_time_cropped = duration_cast<microseconds>(end_time_cropped - start_time_cropped).count() / 1e6;

    cout << "Processing " << fs::path(original_path).filename() << ":" << endl;
    cout << fixed << setprecision(6);
    cout << "Original image processing time: " << process_time_original << " seconds" << endl;
    cout << "Cropped image processing time: " << process_time_cropped << " seconds" << endl;
    cout << "Original area: " << original_results.area_original << " | Cropped area: " << cropped_results.area_original << endl;
    cout << "Original Convex Hull area: " << original_results.area_hull << " | Cropped Convex Hull area: " << cropped_results.area_hull << endl;
    cout << "Original Area ratio: " << original_results.area_ratio << " | Cropped Area ratio: " << cropped_results.area_ratio << endl;
    cout << "Original circularity: " << original_results.circularity_original << " | Cropped circularity: " << cropped_results.circularity_original << endl;
    cout << "Original Convex Hull circularity: " << original_results.circularity_hull << " | Cropped Convex Hull circularity: " << cropped_results.circularity_hull << endl;
    cout << "Original Circularity ratio: " << original_results.circularity_ratio << " | Cropped Circularity ratio: " << cropped_results.circularity_ratio << endl;
    cout << endl;

    // 繪製輪廓
    Mat original_contour_image = Mat::zeros(original_background.size(), CV_8U);
    Mat original_hull_contour_image = Mat::zeros(original_background.size(), CV_8U);
    Mat cropped_contour_image = Mat::zeros(cropped_background.size(), CV_8U);
    Mat cropped_hull_contour_image = Mat::zeros(cropped_background.size(), CV_8U);

    drawContours(original_contour_image, vector<vector<Point>>{original_results.contour}, -1, Scalar(255), 1);
    drawContours(original_hull_contour_image, vector<vector<Point>>{original_results.hull}, -1, Scalar(255), 1);
    drawContours(cropped_contour_image, vector<vector<Point>>{cropped_results.contour}, -1, Scalar(255), 1);
    drawContours(cropped_hull_contour_image, vector<vector<Point>>{cropped_results.hull}, -1, Scalar(255), 1);

    imshow("Original Contour - " + fs::path(original_path).filename().string(), original_contour_image);
    imshow("Original Convex Hull - " + fs::path(original_path).filename().string(), original_hull_contour_image);
    imshow("Cropped Contour - " + fs::path(cropped_path).filename().string(), cropped_contour_image);
    imshow("Cropped Convex Hull - " + fs::path(cropped_path).filename().string(), cropped_hull_contour_image);
    waitKey(0);
    destroyAllWindows();
}

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    cout << "OpenCV version: " << CV_VERSION << endl;

    string original_folder = "Test_images/Slight under focus/";
    string cropped_folder = "Test_images/Cropped/";
    string original_background_path = original_folder + "background.tiff";
    string cropped_background_path = cropped_folder + "background.tiff";

    Mat original_background = imread(original_background_path, IMREAD_GRAYSCALE);
    if (original_background.empty()) {
        cout << "Error: Unable to read original background image: " << original_background_path << endl;
        return -1;
    }

    Mat cropped_background = imread(cropped_background_path, IMREAD_GRAYSCALE);
    if (cropped_background.empty()) {
        cout << "Error: Unable to read cropped background image: " << cropped_background_path << endl;
        return -1;
    }

    for (const auto& entry : fs::directory_iterator(original_folder)) {
        if (entry.path().extension() == ".tiff" && entry.path().filename() != "background.tiff") {
            string original_path = entry.path().string();
            string cropped_path = cropped_folder + entry.path().filename().string();

            if (fs::exists(cropped_path)) {
                process_and_compare(original_path, cropped_path, original_background, cropped_background);
            } else {
                cout << "Cropped image not found for: " << entry.path().filename() << endl;
            }
        }
    }

    return 0;
}
