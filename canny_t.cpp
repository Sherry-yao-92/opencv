#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <numeric>

namespace fs = std::filesystem;

struct ImageData {
    cv::Mat image;
    std::string name;
};

struct ThreadSafeQueue {
    std::queue<ImageData> queue;
    std::mutex mutex;
    std::condition_variable cond;

    void push(ImageData&& data) {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push(std::move(data));
        cond.notify_one();
    }

    bool pop(ImageData& data) {
        std::unique_lock<std::mutex> lock(mutex);
        cond.wait(lock, [this] { return !queue.empty(); });
        if (queue.empty()) return false;
        data = std::move(queue.front());
        queue.pop();
        return true;
    }
};

cv::Mat load_image(const std::string& image_path) {
    return cv::imread(image_path, cv::IMREAD_GRAYSCALE);
}

cv::Mat process_image(const cv::Mat& image, const cv::Mat& blurred_bg) {
    cv::Mat blurred, bg_sub, binary, dilate1, erode1, erode2, dilate2;
    cv::GaussianBlur(image, blurred, cv::Size(5, 5), 0);
    cv::subtract(blurred_bg, blurred, bg_sub);
    cv::threshold(bg_sub, binary, 10, 255, cv::THRESH_BINARY);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    cv::dilate(binary, dilate1, kernel, cv::Point(-1, -1), 2);
    cv::erode(dilate1, erode1, kernel, cv::Point(-1, -1), 2);
    cv::erode(erode1, erode2, kernel, cv::Point(-1, -1), 1);
    cv::dilate(erode2, dilate2, kernel, cv::Point(-1, -1), 1);
    return dilate2;
}

cv::Mat find_contours(const cv::Mat& processed_image) {
    cv::Mat edges, contour_image = cv::Mat::zeros(processed_image.size(), CV_8UC1);
    std::vector<std::vector<cv::Point>> contours;
    cv::Canny(processed_image, edges, 50, 150);
    cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(contour_image, contours, -1, cv::Scalar(255), 1);
    return contour_image;
}

void worker_load(const std::vector<std::string>& image_paths, ThreadSafeQueue& output_queue) {
    for (const auto& path : image_paths) {
        cv::Mat image = load_image(path);
        output_queue.push({image, fs::path(path).filename().string()});
    }
    output_queue.push({cv::Mat(), ""});  // Sentinel
}

void worker_process(ThreadSafeQueue& input_queue, ThreadSafeQueue& output_queue, const cv::Mat& blurred_bg) {
    ImageData data;
    while (input_queue.pop(data) && !data.image.empty()) {
        cv::Mat processed = process_image(data.image, blurred_bg);
        output_queue.push({processed, data.name});
    }
    output_queue.push({cv::Mat(), ""});  // Sentinel
}

void worker_contour(ThreadSafeQueue& input_queue, ThreadSafeQueue& output_queue) {
    ImageData data;
    while (input_queue.pop(data) && !data.image.empty()) {
        cv::Mat contour_image = find_contours(data.image);
        output_queue.push({contour_image, data.name});
    }
    output_queue.push({cv::Mat(), ""});  // Sentinel
}

double calculate_circularity(const std::vector<cv::Point>& contour) {
    double area = cv::contourArea(contour);
    double perimeter = cv::arcLength(contour, true);
    return 4 * CV_PI * area / (perimeter * perimeter);
}

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    std::string directory = "Test_images/Slight under focus";
    std::string background_path = directory + "/background.tiff";

    std::vector<std::string> image_paths;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.path().extension() == ".tiff" && entry.path().filename() != "background.tiff") {
            image_paths.push_back(entry.path().string());
        }
    }

    cv::Mat background = load_image(background_path);
    cv::Mat blurred_bg;
    cv::GaussianBlur(background, blurred_bg, cv::Size(5, 5), 0);

    std::vector<ThreadSafeQueue> queues(4);

    std::thread t1(worker_load, std::ref(image_paths), std::ref(queues[1]));
    std::thread t2(worker_process, std::ref(queues[1]), std::ref(queues[2]), std::ref(blurred_bg));
    std::thread t3(worker_contour, std::ref(queues[2]), std::ref(queues[3]));

    auto start_time = std::chrono::high_resolution_clock::now();

    t1.join();
    t2.join();
    t3.join();

    std::vector<std::tuple<std::string, cv::Mat, cv::Mat, cv::Mat, double, double, double>> results;
    std::vector<double> processing_times;
    ImageData data;
    while (queues[3].pop(data) && !data.image.empty()) {
        auto start_time = std::chrono::high_resolution_clock::now();

        cv::Mat original_contour = cv::Mat::zeros(data.image.size(), CV_8UC1);
        cv::Mat hull_contour = cv::Mat::zeros(data.image.size(), CV_8UC1);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(data.image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if (!contours.empty()) {
            auto largest_contour = *std::max_element(contours.begin(), contours.end(),
                [](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2) {
                    return cv::contourArea(c1) < cv::contourArea(c2);
                });

            std::vector<cv::Point> hull;
            cv::convexHull(largest_contour, hull);

            double circularity = calculate_circularity(largest_contour);
            double hull_circularity = calculate_circularity(hull);

            cv::drawContours(original_contour, std::vector<std::vector<cv::Point>>{largest_contour}, -1, cv::Scalar(255), 1);
            cv::drawContours(hull_contour, std::vector<std::vector<cv::Point>>{hull}, -1, cv::Scalar(255), 1);

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            double processing_time = duration.count() / 1e6;
            processing_times.push_back(processing_time);

            results.push_back(std::make_tuple(data.name, data.image, original_contour, hull_contour, circularity, hull_circularity, processing_time));
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    std::cout << "Total execution time: " << total_duration.count() / 1e6 << " seconds" << std::endl;

    double avg_processing_time = std::accumulate(processing_times.begin(), processing_times.end(), 0.0) / processing_times.size();
    std::cout << "Average processing time per image: " << avg_processing_time << " seconds" << std::endl;

    for (const auto& [name, contour_image, original_contour, hull_contour, circularity, hull_circularity, processing_time] : results) {
        std::cout << "Image: " << name << std::endl;
        std::cout << "Processing time: " << processing_time << " seconds" << std::endl;
        std::cout << "Circularity: " << circularity << std::endl;
        std::cout << "Hull Circularity: " << hull_circularity << std::endl;
        std::cout << "Circularity Ratio: " << hull_circularity / circularity << std::endl;
        std::cout << std::endl;

        cv::imshow("Original Contour: " + name, original_contour);
        cv::imshow("Convex Hull: " + name, hull_contour);
        std::cout << "Showing contours for image: " << name << ". Press any key to continue..." << std::endl;
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    return 0;
}