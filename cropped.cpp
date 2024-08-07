#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

cv::Rect findCropRegion(const cv::Mat& originalImage, const cv::Mat& croppedImage) {
    cv::Mat result;
    cv::matchTemplate(originalImage, croppedImage, result, cv::TM_SQDIFF_NORMED);

    double minVal;
    cv::Point minLoc;
    cv::minMaxLoc(result, &minVal, nullptr, &minLoc, nullptr);

    return cv::Rect(minLoc.x, minLoc.y, croppedImage.cols, croppedImage.rows);
}

int main() {
    std::string originalFolder = "Test_images/Slight under focus";
    std::string croppedFolder = "Test_images/Cropped";
    
    // 讀取手動裁切的圖像
    cv::Mat croppedReference = cv::imread(croppedFolder + "/0000.tiff");
    if (croppedReference.empty()) {
        std::cerr << "Error: Could not load cropped reference image" << std::endl;
        return -1;
    }

    // 讀取原始的未裁切圖像
    cv::Mat originalReference = cv::imread(originalFolder + "/0000.tiff");
    if (originalReference.empty()) {
        std::cerr << "Error: Could not load original reference image" << std::endl;
        return -1;
    }

    // 找出裁切區域
    cv::Rect cropRegion = findCropRegion(originalReference, croppedReference);

    // 處理其他圖像
    for (const auto& entry : fs::directory_iterator(originalFolder)) {
        if (entry.path().extension() == ".tiff" && entry.path().filename() != "0000.tiff") {
            std::string inputFilePath = entry.path().string();
            std::string outputFilePath = croppedFolder + "/" + entry.path().filename().string();

            // 讀取原始圖像
            cv::Mat image = cv::imread(inputFilePath);
            if (image.empty()) {
                std::cerr << "Error: Could not load image '" << inputFilePath << "'" << std::endl;
                continue;
            }

            // 確保裁切區域不超出圖像邊界
            cv::Rect safeRegion = cropRegion & cv::Rect(0, 0, image.cols, image.rows);

            // 裁切圖像
            cv::Mat croppedImage = image(safeRegion);

            // 儲存裁切後的圖像
            if (!cv::imwrite(outputFilePath, croppedImage)) {
                std::cerr << "Error: Could not save cropped image to '" << outputFilePath << "'" << std::endl;
            } else {
                std::cout << "Saved cropped image to '" << outputFilePath << "'" << std::endl;
            }
        }
    }

    return 0;
}
