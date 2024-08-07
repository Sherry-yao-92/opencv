#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <thread>
#include <mutex>
#include <atomic>
#include <numeric>

#define _USE_MATH_DEFINES
#include <math.h>

namespace fs = std::filesystem;

using namespace cv;
using namespace std;
using namespace std::chrono;

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

ContourMetrics process_image(const string& img_path, const Mat& background) {
    Mat img = imread(img_path, IMREAD_GRAYSCALE);
    if (img.empty()) {
        cout << "Error: Unable to read image: " << img_path << endl;
        return ContourMetrics();
    }

    Mat blur_img, blur_background;
    // 並行執行 GaussianBlur
    thread t1([&]() { GaussianBlur(img, blur_img, Size(3, 3), 0); });
    thread t2([&]() { GaussianBlur(background, blur_background, Size(3, 3), 0); });
    t1.join();
    t2.join();

    Mat substract;
    subtract(blur_background, blur_img, substract);

    Mat binary;
    threshold(substract, binary, 10, 255, THRESH_BINARY);

    Mat kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));

    Mat erode1, dilate1, dilate2, erode2;
    // 並行執行形態學操作
    thread t3([&]() {
        erode(binary, erode1, kernel);
        dilate(erode1, dilate1, kernel);
    });
    thread t4([&]() {
        Mat temp;
        dilate(binary, temp, kernel);
        erode(temp, erode2, kernel);
    });
    t3.join();
    t4.join();

    dilate(dilate1, dilate2, kernel);

    Mat edge;
    Canny(erode2, edge, 50, 150);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(edge, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    return calculate_contour_metrics(contours);
}

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    cout << "OpenCV version: " << CV_VERSION << endl;

    string img_folder = "Test_images/Slight under focus/";
    string background_path = img_folder + "background.tiff";

    Mat background = imread(background_path, IMREAD_GRAYSCALE);
    if (background.empty()) {
        cout << "Error: Unable to read background image: " << background_path << endl;
        return -1;
    }

    vector<string> image_paths;
    for (const auto& entry : fs::directory_iterator(img_folder)) {
        if (entry.path().extension() == ".tiff" && entry.path().filename() != "background.tiff") {
            image_paths.push_back(entry.path().string());
        }
    }

    vector<double> processing_times;
    vector<ContourMetrics> all_results(image_paths.size());
    mutex results_mutex;
    atomic<int> processed_count(0);

    // 並行處理多張圖像
    auto process_images = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            auto start_time = high_resolution_clock::now();
            ContourMetrics results = process_image(image_paths[i], background);
            auto end_time = high_resolution_clock::now();

            auto process_time = duration_cast<microseconds>(end_time - start_time).count() / 1e6;

            {
                lock_guard<mutex> lock(results_mutex);
                processing_times.push_back(process_time);
                all_results[i] = results;
            }

            processed_count++;
            cout << "Processed " << processed_count << " of " << image_paths.size() << " images\r" << flush;
        }
    };

    int num_threads = thread::hardware_concurrency();
    vector<thread> threads;
    int chunk_size = image_paths.size() / num_threads;
    for (int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? image_paths.size() : (i + 1) * chunk_size;
        threads.emplace_back(process_images, start, end);
    }

    for (auto& t : threads) {
        t.join();
    }

    cout << endl;

    double total_time = std::accumulate(processing_times.begin(), processing_times.end(), 0.0);
    double avg_time = total_time / processing_times.size();

    cout << "Total processing time: " << total_time << " seconds" << endl;
    cout << "Average processing time per image: " << avg_time << " seconds" << endl;

    for (size_t i = 0; i < image_paths.size(); ++i) {
        const auto& img_path = image_paths[i];
        const auto& results = all_results[i];
        double process_time = processing_times[i];

        cout << "Processing " << fs::path(img_path).filename() << ":" << endl;
        cout << fixed << setprecision(6);
        cout << "Processing time: " << process_time << " seconds" << endl;
        cout << "Original area: " << results.area_original << endl;
        cout << "Convex Hull area: " << results.area_hull << endl;
        cout << "Area ratio (hull/original): " << results.area_ratio << endl;
        cout << "Original circularity: " << results.circularity_original << endl;
        cout << "Convex Hull circularity: " << results.circularity_hull << endl;
        cout << "Circularity ratio (hull/original): " << results.circularity_ratio << endl;
        cout << endl;

        Mat original_contour_image = Mat::zeros(background.size(), CV_8U);
        Mat hull_contour_image = Mat::zeros(background.size(), CV_8U);

        if (!results.contour.empty()) {
            drawContours(original_contour_image, vector<vector<Point>>{results.contour}, -1, Scalar(255), 1);
            drawContours(hull_contour_image, vector<vector<Point>>{results.hull}, -1, Scalar(255), 1);

            imshow("Original Contour - " + fs::path(img_path).filename().string(), original_contour_image);
            imshow("Convex Hull - " + fs::path(img_path).filename().string(), hull_contour_image);
            waitKey(0);
            destroyAllWindows();
        } else {
            cout << "No contours found for this image." << endl;
        }
    }

    return 0;
}