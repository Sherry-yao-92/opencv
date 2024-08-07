#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
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
ContourMetrics results = {};
if (contours.empty()) {
return results;
}

```
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

```

}

ContourMetrics process_image(const string& img_path, const cuda::GpuMat& background_gpu, cuda::Stream& stream) {
Mat img = imread(img_path, IMREAD_GRAYSCALE);
if (img.empty()) {
cout << "Error: Unable to read image: " << img_path << endl;
return ContourMetrics();
}

```
cuda::GpuMat img_gpu, blur_img_gpu, subtract_result_gpu, binary_gpu;
img_gpu.upload(img, stream);

// 使用CUDA进行高斯模糊
Ptr<cuda::Filter> gaussian_filter = cuda::createGaussianFilter(CV_8UC1, CV_8UC1, Size(3, 3), 0);
gaussian_filter->apply(img_gpu, blur_img_gpu, stream);

// 使用CUDA进行图像相减和二值化
cuda::subtract(background_gpu, blur_img_gpu, subtract_result_gpu, noArray(), -1, stream);
cuda::threshold(subtract_result_gpu, binary_gpu, 10, 255, THRESH_BINARY, stream);

// 使用CUDA进行形态学操作
Mat kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));
Ptr<cuda::Filter> morph_filter = cuda::createMorphologyFilter(MORPH_CLOSE, CV_8UC1, kernel);
cuda::GpuMat morph_gpu;
morph_filter->apply(binary_gpu, morph_gpu, stream);

// 将结果下载回CPU进行轮廓检测
Mat morph;
morph_gpu.download(morph, stream);

// 等待stream完成
stream.waitForCompletion();

vector<vector<Point>> contours;
findContours(morph, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

// 找到最大的轮廓
auto max_contour = max_element(contours.begin(), contours.end(),
    [](const vector<Point>& c1, const vector<Point>& c2) {
        return contourArea(c1) < contourArea(c2);
    });

if (max_contour != contours.end()) {
    // 只处理最大的轮廓
    vector<vector<Point>> single_contour = {*max_contour};
    return calculate_contour_metrics(single_contour);
}

return ContourMetrics(); // 如果没有找到轮廓，返回空的结果

```

}

void process_file(const string& img_path, const cuda::GpuMat& background_gpu, vector<double>& processing_times, vector<ContourMetrics>& results_list, cuda::Stream& stream) {
auto start_time = high_resolution_clock::now();
ContourMetrics results = process_image(img_path, background_gpu, stream);
auto end_time = high_resolution_clock::now();

```
auto process_time = duration_cast<microseconds>(end_time - start_time).count() / 1e6;
processing_times.push_back(process_time);
results_list.push_back(results);

```

}

int main() {
cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
cout << "OpenCV version: " << CV_VERSION << endl;

```
std::string directory = "E:/Data/Sherry&peggy/Test_images/Slight under focus";
std::string background_path = "E:/Data/Sherry&peggy/Test_images/Slight under focus/background.tiff";

Mat background = imread(background_path, IMREAD_GRAYSCALE);
if (background.empty()) {
    cout << "Error: Unable to read background image: " << background_path << endl;
    return -1;
}

// 将背景图像上传到GPU
cuda::GpuMat d_background_gpu;
d_background_gpu.upload(background);

// 预先对背景进行高斯模糊
Ptr<cuda::Filter> gaussian_filter = cuda::createGaussianFilter(CV_8UC1, CV_8UC1, Size(3, 3), 0);
gaussian_filter->apply(d_background_gpu, d_background_gpu);

vector<double> processing_times;
vector<ContourMetrics> results_list;
vector<string> img_paths;

for (const auto& entry : fs::directory_iterator(directory)) {
    if (entry.path().extension() == ".tiff" && entry.path().filename() != "background.tiff") {
        img_paths.push_back(entry.path().string());
    }
}

// 创建CUDA streams
vector<cuda::Stream> streams(img_paths.size());

// 处理所有图片并记录时间
for (size_t i = 0; i < img_paths.size(); ++i) {
    process_file(img_paths[i], d_background_gpu, processing_times, results_list, streams[i]);
}

// 计算并显示平均处理时间
if (!processing_times.empty()) {
    double total_time = std::accumulate(processing_times.begin(), processing_times.end(), 0.0);
    double average_time = total_time / processing_times.size();
    cout << "Average processing time: " << fixed << setprecision(6) << average_time << " seconds" << endl;
}

// 显示所有图片和数据
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

```

}