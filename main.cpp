#include <opencv2/opencv.hpp>
#include <iostream>

int main(){
    std::string imagePath = "0000.tiff";
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    cv::imshow("Display window", image);
    cv::waitKey(0);
    return 0;
}