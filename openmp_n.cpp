#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <iomanip>
#include <omp.h>
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

ContourMetrics process_image(const string& img_path, const Mat& background) {
    Mat img = imread(img_path, IMREAD_GRAYSCALE);
    if (img.empty()) {
        cout << "Error: Unable to read image: " << img_path << endl;
        return ContourMetrics(); // 返回一个默认构造的 ContourMetrics 对象
    }

    Mat blur_img, blur_background, substract, binary, erode1, dilate1, dilate2, erode2, edge;

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            GaussianBlur(img, blur_img, Size(3, 3), 0);
        }
        #pragma omp section
        {
            GaussianBlur(background, blur_background, Size(3, 3), 0);
        }
    }

    subtract(blur_background, blur_img, substract);
    threshold(substract, binary, 10, 255, THRESH_BINARY);

    Mat kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            erode(binary, erode1, kernel);
            dilate(erode1, dilate1, kernel);
        }
        #pragma omp section
        {
            dilate(binary, dilate2, kernel);
            erode(dilate2, erode2, kernel);
        }
    }

    Canny(erode2, edge, 50, 150);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(edge, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    return calculate_contour_metrics(contours);
}

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

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

    sort(image_paths.begin(), image_paths.end(), [](const fs::path& a, const fs::path& b) {
        return a.filename() < b.filename();
    });

    map<fs::path, ContourMetrics> results;

    #pragma omp parallel for
    for (int i = 0; i < image_paths.size(); ++i) {
        const auto& img_path = image_paths[i];
        
        auto start_time = high_resolution_clock::now();
        ContourMetrics metrics = process_image(img_path.string(), background);
        auto end_time = high_resolution_clock::now();

        metrics.process_time = duration_cast<microseconds>(end_time - start_time).count();

        #pragma omp critical
        {
            results[img_path] = metrics;
        }
    }

    // 计算平均处理时间
    long long total_time = 0;
    for (const auto& result : results) {
        total_time += result.second.process_time;
    }
    double average_time = static_cast<double>(total_time) / results.size();

    // 打印平均处理时间
    cout << "Average processing time: " << fixed << setprecision(2) << average_time << " microseconds" << endl;
    cout << endl;

    // 按順序輸出結果
    for (const auto& img_path : image_paths) {
        const auto& metrics = results[img_path];

        cout << "Processing " << img_path.filename() << ":" << endl;
        cout << fixed << setprecision(6);
        cout << "Processing time: " << metrics.process_time << " microseconds" << endl;
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

            imshow("Original Contour - " + img_path.filename().string(), original_contour_image);
            imshow("Convex Hull - " + img_path.filename().string(), hull_contour_image);
            waitKey(0);
            destroyAllWindows();
        } else {
            cout << "No contours found for this image." << endl;
        }
    }

    return 0;
}