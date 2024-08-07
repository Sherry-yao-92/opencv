#include <vector>
#include <stack>
#include <utility>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <opencv2/core/utils/logger.hpp>

std::vector<std::vector<std::pair<int, int>>> trace_contours(const std::vector<std::vector<int>>& edge_image) {
    std::vector<std::vector<std::pair<int, int>>> contours;
    std::vector<std::vector<bool>> visited(edge_image.size(), std::vector<bool>(edge_image[0].size(), false));
    std::vector<std::pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};  // 4-connectivity

    for (int i = 0; i < edge_image.size(); ++i) {
        for (int j = 0; j < edge_image[0].size(); ++j) {
            if (edge_image[i][j] != 0 && !visited[i][j]) {
                std::vector<std::pair<int, int>> contour;
                std::stack<std::pair<int, int>> stack;
                stack.push({i, j});

                while (!stack.empty()) {
                    auto [x, y] = stack.top();
                    stack.pop();

                    if (!visited[x][y]) {
                        visited[x][y] = true;
                        contour.push_back({x, y});

                        for (const auto& [dx, dy] : directions) {
                            int xn = x + dx, yn = y + dy;
                            if (xn >= 0 && xn < edge_image.size() && yn >= 0 && yn < edge_image[0].size()) {
                                if (edge_image[xn][yn] != 0 && !visited[xn][yn]) {
                                    stack.push({xn, yn});
                                }
                            }
                        }
                    }
                }

                if (!contour.empty()) {
                    contours.push_back(contour);
                }
            }
        }
    }

    return contours;
}

void process_image(const std::string& image_path, const std::string& background_path) {
    // 讀取圖像
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    cv::Mat background = cv::imread(background_path, cv::IMREAD_GRAYSCALE);
    
    if (image.empty() || background.empty()) {
        std::cerr << "Error: Unable to read image files." << std::endl;
        return;
    }

    cv::Mat blurred_bg;
    cv::GaussianBlur(background, blurred_bg, cv::Size(5, 5), 0);
    //cv::imshow("raw", image);
    
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

    auto start_time = std::chrono::high_resolution_clock::now();

    // 應用高斯模糊
    cv::Mat blurred;
    cv::GaussianBlur(image, blurred, cv::Size(5, 5), 0);
    //cv::imshow("blurred", blurred);

    // 背景減除
    std::cout << "Blurred shape: " << blurred.size() << ", Blurred BG shape: " << blurred_bg.size() << std::endl;
    cv::Mat bg_sub;
    cv::subtract(blurred_bg, blurred, bg_sub);
    //cv::imshow("bg_sub", bg_sub);

    // 應用閾值
    cv::Mat binary;
    cv::threshold(bg_sub, binary, 10, 255, cv::THRESH_BINARY);
    //cv::imshow("binary", binary);

    // 膨脹和腐蝕以去除噪聲
    cv::Mat dilate1, erode1, erode2, dilate2;
    cv::dilate(binary, dilate1, kernel, cv::Point(-1,-1), 2);
    //cv::imshow("dilate1", dilate1);
    cv::erode(dilate1, erode1, kernel, cv::Point(-1,-1), 2);
    //cv::imshow("erode1", erode1);
    cv::erode(erode1, erode2, kernel, cv::Point(-1,-1), 1);
    //cv::imshow("erode2", erode2);
    cv::dilate(erode2, dilate2, kernel, cv::Point(-1,-1), 1);
    //cv::imshow("dilate2", dilate2);

    // 應用Canny邊緣檢測器
    cv::Mat edges;
    cv::Canny(dilate2, edges, 50, 150);
    // cv::imshow("canny edges", edges);

    // 將 cv::Mat 轉換為 std::vector<std::vector<int>>
    std::vector<std::vector<int>> edge_image(edges.rows, std::vector<int>(edges.cols));
    for (int i = 0; i < edges.rows; ++i) {
        for (int j = 0; j < edges.cols; ++j) {
            edge_image[i][j] = edges.at<uchar>(i, j);
        }
    }

    // 追踪輪廓
    auto contours = trace_contours(edge_image);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Processing time: " << duration.count() << " ms" << std::endl;

    // 準備一個圖像來繪製輪廓
    cv::Mat contour_image = cv::Mat::zeros(image.size(), CV_8UC1);

    // 繪製每個輪廓
    for (const auto& contour : contours) {
        for (const auto& point : contour) {
            if (point.first >= 0 && point.first < contour_image.rows && 
                point.second >= 0 && point.second < contour_image.cols) {
                contour_image.at<uchar>(point.first, point.second) = 255;
            }
        }
    }

    // 顯示結果圖像
    cv::imshow("Processed Image", contour_image);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    process_image("Test_images/Slight under focus/0066.tiff", "Test_images/Slight under focus/background.tiff");
    return 0;
}