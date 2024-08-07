import cv2
import numpy as np
import os
import time
import threading
from queue import Queue,Empty


class ContourMetrics:
    def __init__(self):
        self.area_original = 0
        self.area_hull = 0
        self.area_ratio = 0
        self.circularity_original = 0
        self.circularity_hull = 0
        self.circularity_ratio = 0

def calculate_contour_metrics(contours):
    #if not contours:
    #    print("No contours were found.")
    #    return None

    cnt = max(contours, key=cv2.contourArea)
    area_original = cv2.contourArea(cnt)
    perimeter_original = cv2.arcLength(cnt, True)

    if area_original <= 1e-6 or perimeter_original <= 1e-6:
        print(f"Invalid contour measurements: area={area_original}, perimeter={perimeter_original}")
        return 

    circularity_original = 4 * np.pi * area_original / (perimeter_original ** 2)

    hull = cv2.convexHull(cnt)
    area_hull = cv2.contourArea(hull)
    perimeter_hull = cv2.arcLength(hull, True)

    circularity_hull = 4 * np.pi * area_hull / (perimeter_hull ** 2)

    results = ContourMetrics()
    results.area_original = area_original
    results.area_hull = area_hull
    results.area_ratio = area_hull / area_original if area_original > 0 else 0
    results.circularity_original = circularity_original
    results.circularity_hull = circularity_hull
    results.circularity_ratio = circularity_hull / circularity_original if circularity_original > 0 else 0

    return results

def process_single_image(image_path, blurred_bg):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("false to load image.")
            return 

        start_time = time.perf_counter()

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        bg_sub = cv2.subtract(blurred_bg, blurred)
        _, binary = cv2.threshold(bg_sub, 10, 255, cv2.THRESH_BINARY)
        dilate1 = cv2.dilate(binary, kernel, iterations=2)
        erode1 = cv2.erode(dilate1, kernel, iterations=3)
        dilate2 = cv2.dilate(erode1, kernel, iterations=1)

        edges = cv2.Canny(dilate2, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        if not contours:
            print("no contours were found.")
            return 
        
        metrics = calculate_contour_metrics(contours)
        if metrics is None:
            return
        
        end_time = time.perf_counter()
        duration = (end_time - start_time) * 1e6  # Convert to milliseconds
        return metrics, duration
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return 

def worker(image_queue, result_queue, blurred_bg, lock):
    while True:
        path = image_queue.get()
        if path is None:
            break
        
        result = process_single_image(path, blurred_bg)
        if result:
            metrics, duration = result
            with lock:
                result_queue.put((os.path.basename(path), metrics, duration))
        else:
            with lock:
                print(f"Skipping invalid result for {path}")
        image_queue.task_done()
        

def main():
    directory = "Test_images/Slight under focus"
    background_path = os.path.join(directory, "background.tiff")
    background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
    blurred_bg = cv2.GaussianBlur(background, (5, 5), 0)

    image_queue = Queue()
    result_queue = Queue()
    lock = threading.Lock()
    
    # Start worker threads
    num_threads = os.cpu_count()
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker, args=(image_queue, result_queue, blurred_bg, lock))
        t.start()
        threads.append(t)
    
    # Enqueue tasks and collect results
    for entry in os.scandir(directory):
        if entry.name.endswith('.tiff') and entry.name != 'background.tiff':
            image_queue.put(entry.path)
        
        # Collect and print results
        while not result_queue.empty():
            try:
                filename, metrics, duration= result_queue.get_nowait()
                print(f"File: {filename}")
                print(f"Duration: {duration:.2f} microseconds")
                print(f"Area Original: {metrics.area_original:.2f}")
                print(f"Area Hull: {metrics.area_hull:.2f}")
                print(f"Area Ratio: {metrics.area_ratio:.2f}")
                print(f"Circularity Original: {metrics.circularity_original:.2f}")
                print(f"Circularity Hull: {metrics.circularity_hull:.2f}")
                print(f"Circularity Ratio: {metrics.circularity_ratio:.2f}")
                print()
            except Empty:
                pass
    
    # Add sentinel values to stop workers
    for _ in range(num_threads):
        image_queue.put(None)
                
    # Wait for all tasks to complete
    for t in threads:
        t.join()
        
    # Collect remaining results
    while True:
        try:
            filename, metrics, duration = result_queue.get(timeout=1)
            with lock:
                print(f"File: {filename}")
                print(f"Duration: {duration:.2f} microseconds")
                print(f"Area Original: {metrics.area_original:.2f}")
                print(f"Area Hull: {metrics.area_hull:.2f}")
                print(f"Area Ratio: {metrics.area_ratio:.2f}")
                print(f"Circularity Original: {metrics.circularity_original:.2f}")
                print(f"Circularity Hull: {metrics.circularity_hull:.2f}")
                print(f"Circularity Ratio: {metrics.circularity_ratio:.2f}")
                print()
        except Empty:
            break

    # Wait for all tasks to complete
    for t in threads:
        t.join()

if __name__ == "__main__":
    main()
