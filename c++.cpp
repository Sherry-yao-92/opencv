#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>
#include <tbb/concurrent_queue.h>
#include <atomic>
#include <mutex>

#define _USE_MATH_DEFINES
#include <math.h>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

struct ContourMetrics {
    double area_original = 0;
    double area_hull = 0;
    double area_ratio = 0;
    double circularity_original = 0;
    double circularity_hull = 0;
    double circularity_ratio = 0;
};

ContourMetrics calculate_contour_metrics(const vector<vector<Point>>& contours) {
    if (contours.empty()) {
        return ContourMetrics();
    }

    auto cnt = *max_element(contours.begin(), contours.end(),
                            [](const auto& c1, const auto& c2) { return contourArea(c1) < contourArea(c2); });

    double area_original = contourArea(cnt);
    double perimeter_original = arcLength(cnt, true);

    if (area_original <= 1e-6 || perimeter_original <= 1e-6) {
        return ContourMetrics();
    }

    double circularity_original = 4 * M_PI * area_original / (perimeter_original * perimeter_original);

    vector<Point> hull;
    convexHull(cnt, hull);

    double area_hull = contourArea(hull);
    double perimeter_hull = arcLength(hull, true);

    if (area_hull <= 1e-6 || perimeter_hull <= 1e-6) {
        return ContourMetrics();
    }

    double circularity_hull = 4 * M_PI * area_hull / (perimeter_hull * perimeter_hull);

    ContourMetrics results;
    results.area_original = area_original;
    results.area_hull = area_hull;
    results.area_ratio = area_hull / area_original;
    results.circularity_original = circularity_original;
    results.circularity_hull = circularity_hull;
    results.circularity_ratio = circularity_hull / circularity_original;

    return results;
}

void process_single_image(const string& image_path, const Mat& blurred_bg, vector<vector<Point>>& contours, ContourMetrics& metrics, double& duration) {
    Mat image = imread(image_path, IMREAD_GRAYSCALE);
    auto start_time = chrono::high_resolution_clock::now();

    Mat kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));
    Mat blurred;
    GaussianBlur(image, blurred, Size(5, 5), 0);
    Mat bg_sub;
    subtract(blurred_bg, blurred, bg_sub);
    Mat binary;
    threshold(bg_sub, binary, 10, 255, THRESH_BINARY);

    Mat dilate1, erode1, dilate2;
    dilate(binary, dilate1, kernel, Point(-1, -1), 2);
    erode(dilate1, erode1, kernel, Point(-1, -1), 3);
    dilate(erode1, dilate2, kernel, Point(-1, -1), 1);

    Mat edges;
    Canny(dilate2, edges, 50, 150);

    vector<Vec4i> hierarchy;
    findContours(edges, contours, hierarchy, RETR_LIST, CHAIN_APPROX_NONE);

    if (contours.empty()) {
        return;
    }

    auto end_time = chrono::high_resolution_clock::now();
    duration = chrono::duration<double, micro>(end_time - start_time).count();

    metrics = calculate_contour_metrics(contours);
}

void run_experiment(string directory, vector<pair<double, double>>& results) {
    string background_path = directory + "/background.tiff";
    Mat background = imread(background_path, IMREAD_GRAYSCALE);
    Mat blurred_bg;
    GaussianBlur(background, blurred_bg, Size(5, 5), 0);

    atomic<double> average_time(0);
    atomic<int> number(0);
    atomic<double> max_process_time(0);
    mutex mtx;

    tbb::task_arena arena(thread::hardware_concurrency());
    tbb::task_group group;
    tbb::concurrent_queue<fs::path> image_queue;

    atomic<bool> processing_complete(false);

    arena.execute([&]() {
        group.run([&]() {
            while (!processing_complete || !image_queue.empty()) {
                fs::path path;
                if (image_queue.try_pop(path)) {
                    vector<vector<Point>> contours;
                    ContourMetrics metrics;
                    double process_time;
                    process_single_image(path.string(), blurred_bg, contours, metrics, process_time);

                    lock_guard<mutex> lock(mtx);
                    average_time = average_time + process_time;
                    if (process_time > max_process_time) {
                        max_process_time = process_time;
                    }
                    number++;
                } else {
                    this_thread::yield();
                }
            }
        });
    });

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.path().extension() == ".tiff" && entry.path().filename() != "background.tiff") {
            image_queue.push(entry.path());
        }
    }

    processing_complete = true;
    group.wait();

    results.push_back({ max_process_time.load(), average_time.load() / number.load() });
}

int main() {
    string directory = "Test_images/Slight under focus";
    vector<pair<double, double>> results;

    for (int i = 0; i < 100; ++i) {
        run_experiment(directory, results);
    }

    ofstream file("image_processing_results.csv");
    file << "Max time (C++),Avg time (C++)\n";
    for (const auto& result : results) {
        file << result.first << "," << result.second << "\n";
    }
    file.close();

    return 0;
}
