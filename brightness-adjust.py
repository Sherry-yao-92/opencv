import cv2
import numpy as np
import math
import time

def adjust_brightness(image, target_brightness):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    current_brightness = gray.mean()
    brightness_factor = target_brightness / current_brightness
    image_float = image.astype("float32")
    adjusted_image = image_float * brightness_factor
    adjusted_image = np.clip(adjusted_image, 0, 255).astype("uint8")
    return adjusted_image

def calculate_contour_metrics(contours):
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    
    area_original = cv2.contourArea(cnt)
    perimeter_original = cv2.arcLength(cnt, True)
    circularity_original = float(2 * math.sqrt((math.pi) * area_original)) / perimeter_original
    
    hull = cv2.convexHull(cnt)
    
    area_hull = cv2.contourArea(hull)
    perimeter_hull = cv2.arcLength(hull, True)
    circularity_hull = float(2 * math.sqrt((math.pi) * area_hull)) / perimeter_hull
    
    area_ratio = area_hull / area_original
    circularity_ratio = circularity_hull / circularity_original

    return {
        "area_original": area_original,
        "area_hull": area_hull,
        "area_ratio": area_ratio,
        "circularity_original": circularity_original,
        "circularity_hull": circularity_hull,
        "circularity_ratio": circularity_ratio,
        "contour": cnt,
        "hull": hull
    }

def process_image(img, background_path):
    background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)

    start_time = time.perf_counter()

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    blur_background = cv2.GaussianBlur(background, (3, 3), 0)
    substract = cv2.subtract(blur_background, blur_img)
    ret, binary = cv2.threshold(substract, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    erode1 = cv2.erode(binary, kernel)
    dilate1 = cv2.dilate(erode1, kernel)
    dilate2 = cv2.dilate(dilate1, kernel)
    erode2 = cv2.erode(dilate2, kernel)

    edge = cv2.Canny(erode2, 50, 150)
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    results = calculate_contour_metrics(contours)

    end_time = time.perf_counter()
    total_time = end_time - start_time

    return total_time, results

def main():
    img_path = 'Test_images/Slight under focus/new_image.tiff'
    background_path = 'Test_images/Slight under focus/background.tiff'
    output_path = 'C:/Users/USER/HK/0710/Test_images/Slight under focus/adjusted_image.tiff'

    # Read original image
    original_img = cv2.imread(img_path)
    if original_img is None:
        print(f"Error: Unable to read image from {img_path}")
        return

    # Calculate and print original brightness
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    current_brightness = gray.mean()
    print(f"The current brightness of the image is: {current_brightness:.6f}")

    # Adjust brightness
    target_brightness = 109.79
    adjusted_img = adjust_brightness(original_img, target_brightness)

    # Calculate and print adjusted brightness
    adjusted_gray = cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2GRAY)
    adjusted_brightness = adjusted_gray.mean()
    print(f"The adjusted brightness of the image is: {adjusted_brightness:.6f}")

    # Save adjusted image
    cv2.imwrite(output_path, adjusted_img)

    # Process the original and adjusted images
    original_time, original_results = process_image(original_img, background_path)
    adjusted_time, adjusted_results = process_image(adjusted_img, background_path)

    print(f"Original image processing time: {original_time:.6f} seconds")
    print(f"Adjusted image processing time: {adjusted_time:.6f} seconds")

    # Print results for original image
    if original_results:
        print("\nOriginal Image Results:")
        print(f"Original area: {original_results['area_original']:.2f}")
        print(f"Convex Hull area: {original_results['area_hull']:.2f}")
        print(f"Area ratio (hull/original): {original_results['area_ratio']:.6f}")
        print(f"Original circularity: {original_results['circularity_original']:.6f}")
        print(f"Convex Hull circularity: {original_results['circularity_hull']:.6f}")
        print(f"Circularity ratio (hull/original): {original_results['circularity_ratio']:.6f}")

    # Print results for adjusted image
    if adjusted_results:
        print("\nAdjusted Image Results:")
        print(f"Original area: {adjusted_results['area_original']:.2f}")
        print(f"Convex Hull area: {adjusted_results['area_hull']:.2f}")
        print(f"Area ratio (hull/original): {adjusted_results['area_ratio']:.6f}")
        print(f"Original circularity: {adjusted_results['circularity_original']:.6f}")
        print(f"Convex Hull circularity: {adjusted_results['circularity_hull']:.6f}")
        print(f"Circularity ratio (hull/original): {adjusted_results['circularity_ratio']:.6f}")

    # Display images
    cv2.imshow('Original Image', original_img)
    cv2.imshow('Adjusted Image', adjusted_img)

    if original_results:
        original_contour_image = np.zeros_like(original_img)
        original_hull_image = np.zeros_like(original_img)
        cv2.drawContours(original_contour_image, [original_results['contour']], -1, (255,255,255), 1)
        cv2.drawContours(original_hull_image, [original_results['hull']], -1, (255,255,255), 1)
        cv2.imshow('Original Contour', original_contour_image)
        cv2.imshow('Original Convex Hull', original_hull_image)

    if adjusted_results:
        adjusted_contour_image = np.zeros_like(adjusted_img)
        adjusted_hull_image = np.zeros_like(adjusted_img)
        cv2.drawContours(adjusted_contour_image, [adjusted_results['contour']], -1, (255,255,255), 1)
        cv2.drawContours(adjusted_hull_image, [adjusted_results['hull']], -1, (255,255,255), 1)
        cv2.imshow('Adjusted Contour', adjusted_contour_image)
        cv2.imshow('Adjusted Convex Hull', adjusted_hull_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()