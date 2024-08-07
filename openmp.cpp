#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <iomanip>
#include <omp.h>

#define _USE_MATH_DEFINES
#include <math.h>

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
    cout << "OpenCV version: " << CV_VERSION << endl;

    string img_folder = "Test_images\\Slight under focus\\";
    string background_path = img_folder + "background.tiff";
    string img_path = img_folder + "0066.tiff";

    Mat background = imread(background_path, IMREAD_GRAYSCALE);
    if (background.empty()) {
        cout << "Error: Unable to read background image: " << background_path << endl;
        return -1;
    }

    auto start_time = high_resolution_clock::now();
    ContourMetrics results = process_image(img_path, background);
    auto end_time = high_resolution_clock::now();

    auto process_time = duration_cast<microseconds>(end_time - start_time).count() / 1e6;

    cout << "Processing 0066.tiff:" << endl;
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

        imshow("Original Contour - 0066.tiff", original_contour_image);
        imshow("Convex Hull - 0066.tiff", hull_contour_image);
        waitKey(0);
        destroyAllWindows();
    } else {
        cout << "No contours found for this image." << endl;
    }

    return 0;
}