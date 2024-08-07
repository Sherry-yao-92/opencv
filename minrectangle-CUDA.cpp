#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <cstdio> // 含C语言printf

struct ContourMetrics {
    double area_original;
    double area_hull;
    double area_ratio;
    double circularity_original;
    double circularity_hull;
    double circularity_ratio;
};

ContourMetrics calculate_contour_metrics(const vector<vector<Point>>& contours) {
    ContourMetrics results = {}; // 初始化所有成员为0

    if (contours.empty()) {
        printf("No contours to calculate metrics.\n");
        return results; // 返回初始化的结果
    }

    auto cnt = *max_element(contours.begin(), contours.end(),
        [](const auto& c1, const auto& c2) { return contourArea(c1) < contourArea(c2); });

    double area_original = contourArea(cnt);
    double perimeter_original = arcLength(cnt, true);

    if (area_original <= 1e-6 || perimeter_original <= 1e-6) {
        printf("Invalid contour measurements: area=%f, perimeter=%f\n", area_original, perimeter_original);
        return results; // 返回初始化的结果
    }

    double circularity_original = 2 * sqrt(M_PI * area_original) / perimeter_original;

    // 凸包处理
    vector<Point> hull;
    convexHull(cnt, hull);

    double area_hull = contourArea(hull);
    double perimeter_hull = arcLength(hull, true);

    if (area_hull <= 1e-6 || perimeter_hull <= 1e-6) {
        printf("Invalid hull measurements: area=%f, perimeter=%f\n", area_hull, perimeter_hull);
        return results; // 返回初始化的结果
    }

    double circularity_hull = 2 * sqrt(M_PI * area_hull) / perimeter_hull;

    results.area_original = area_original;
    results.area_hull = area_hull;
    results.area_ratio = area_hull / area_original;
    results.circularity_original = circularity_original;
    results.circularity_hull = circularity_hull;
    results.circularity_ratio = circularity_hull / circularity_original;

    return results;
}

void process_image_cropped(const string& image_path, const cuda::GpuMat& d_blurred_bg, vector<vector<Point>>& contours, ContourMetrics& metrics, double& duration) {
    Mat image = imread(image_path, IMREAD_GRAYSCALE);
    if (image.empty()) {
        printf("Error: Unable to read image: %s\n", image_path.c_str());
        return;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    cuda::GpuMat d_image, d_blurred, d_bg_sub, d_binary;
    d_image.upload(image);

    // 高斯模糊
    Ptr<cuda::Filter> gaussianFilter = cuda::createGaussianFilter(d_image.type(), -1, Size(5, 5), 0);
    gaussianFilter->apply(d_image, d_blurred);

    // 背景减除
    cuda::subtract(d_blurred_bg, d_blurred, d_bg_sub);

    // 二值化
    cuda::threshold(d_bg_sub, d_binary, 10, 255, THRESH_BINARY);

    Mat binary;
    d_binary.download(binary);

    vector<vector<Point>> contoursbi;
    vector<Vec4i> hierarchybi;
    cv::findContours(binary, contoursbi, hierarchybi, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    if (contoursbi.empty()) {
        printf("No contours found in the image.\n");
        return;
    } else if (contoursbi.size() > 1) {
        printf("More than one contour found. Exiting.\n");
        return;
    }

    Rect bounding_box = boundingRect(contoursbi[0]);
    int padding = 10;
    bounding_box.x = std::max(0, bounding_box.x - padding);
    bounding_box.y = std::max(0, bounding_box.y - padding);
    bounding_box.width = std::min(binary.cols - bounding_box.x, bounding_box.width + 2 * padding);
    bounding_box.height = std::min(binary.rows - bounding_box.y, bounding_box.height + 2 * padding);

    Mat cropped = binary(bounding_box).clone();
    cuda::GpuMat d_cropped;
    d_cropped.upload(cropped);

    Mat kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));
    Ptr<cuda::Filter> morphologyFilter = cuda::createMorphologyFilter(MORPH_DILATE, d_cropped.type(), kernel);

    morphologyFilter->apply(d_cropped, d_blurred); // 使用 d_blurred 作为输出
    morphologyFilter = cuda::createMorphologyFilter(MORPH_ERODE, d_blurred.type(), kernel);
    morphologyFilter->apply(d_blurred, d_bg_sub); // 使用 d_bg_sub 作为输出
    morphologyFilter = cuda::createMorphologyFilter(MORPH_DILATE, d_bg_sub.type(), kernel);
    morphologyFilter->apply(d_bg_sub, d_binary); // 使用 d_binary 作为输出

    cuda::GpuMat d_edges;
    Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector(50, 150);
    canny->detect(d_binary, d_edges);

    Mat edges;
    d_edges.download(edges);

    vector<Vec4i> hierarchy;
    findContours(edges, contours, hierarchy, RETR_LIST, CHAIN_APPROX_NONE);

    if (contours.empty()) {
        printf("No contours found in the image.\n");
        return;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    duration = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
    
    printf("process time: %f us \n", duration);
    metrics = calculate_contour_metrics(contours);
}

int main() {
    std::string directory = "E:/Data/Sherry&peggy/Test_images/Slight under focus";
    std::string background_path = "E:/Data/Sherry&peggy/Test_images/Slight under focus/background.tiff";
    
    Mat background = imread(background_path, IMREAD_GRAYSCALE);
    if (background.empty()) {
        cout << "Error: Unable to read background image: " << background_path << endl;
        return -1; // 或者处理错误
    }

    cuda::GpuMat d_background, d_blurred_bg;
    d_background.upload(background);
    
    Ptr<cuda::Filter> gaussianFilter = cuda::createGaussianFilter(d_background.type(), -1, Size(5, 5), 0);
    gaussianFilter->apply(d_background, d_blurred_bg);

    vector<vector<Point>> contours_cropped;
    ContourMetrics metrics_cropped;
    double averagetime_c = 0, number = 0;
    double max_processtime_c = 0;

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.path().extension() == ".tiff" && entry.path().filename() != "background.tiff") {
            printf("processing %s\n", entry.path().string().c_str());
            contours_cropped.clear();
            metrics_cropped = ContourMetrics();
            double processtime_c;
            process_image_cropped(entry.path().string(), d_blurred_bg, contours_cropped, metrics_cropped, processtime_c);
            averagetime_c += processtime_c;
            if (processtime_c > max_processtime_c) {
                max_processtime_c = processtime_c;
                printf("****************时间久 image %s\n", entry.path().string().c_str());
            }
            number += 1;
        }
    }
    
    if (number > 0) {
        printf("maximum time for cropped image = %f\n", max_processtime_c);
        printf("average time for cropped image = %f\n", averagetime_c / number);
    } else {
        printf("No images processed.\n");
    }

    return 0;
}