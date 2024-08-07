import cv2
import numpy as np
import os
import time
from queue import Queue
from threading import Thread

def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def process_image(image, blurred_bg):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    bg_sub = cv2.subtract(blurred_bg, blurred)
    _, binary = cv2.threshold(bg_sub, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilate1 = cv2.dilate(binary, kernel, iterations=2)
    erode1 = cv2.erode(dilate1, kernel, iterations=2)
    erode2 = cv2.erode(erode1, kernel, iterations=1)
    dilate2 = cv2.dilate(erode2, kernel, iterations=1)
    return dilate2

def find_contours(processed_image):
    edges = cv2.Canny(processed_image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.zeros_like(processed_image)
    cv2.drawContours(contour_image, contours, -1, (255), 1)
    return contour_image

def worker_load(input_queue, output_queue):
    while True:
        image_path = input_queue.get()
        if image_path is None:
            break
        image = load_image(image_path)
        output_queue.put((image, os.path.basename(image_path)))
    output_queue.put(None)
        

def worker_process(input_queue, output_queue, blurred_bg):
    while True:
        item = input_queue.get()
        if item is None:
            break
        image, image_name = item
        processed = process_image(image, blurred_bg)
        output_queue.put((processed, image_name))
    output_queue.put(None)
        

def worker_contour(input_queue, output_queue):
    while True:
        item = input_queue.get()
        if item is None:
            break
        processed, image_name = item
        contour_image = find_contours(processed)
        output_queue.put((contour_image, image_name))
    output_queue.put(None)

def main():
    directory = 'Test_images/Slight under focus'
    background_path = os.path.join(directory, 'background.tiff')
    files = [f for f in os.listdir(directory) if f.endswith('.tiff') and f != 'background.tiff']
    blurred_bg = cv2.GaussianBlur(load_image(background_path), (5, 5), 0)
    
    queues = [Queue() for _ in range(4)]

    threads = [
        Thread(target=worker_load, args=(queues[0], queues[1])),
        Thread(target=worker_process, args=(queues[1], queues[2], blurred_bg)),
        Thread(target=worker_contour, args=(queues[2], queues[3]))
    ]

    for thread in threads:
        thread.start()

    start_time = time.time()

    for image in files:
        queues[0].put(os.path.join(directory, image))
    queues[0].put(None)
    
    for thread in threads:
        thread.join()

    results = {}
    while True:
        item = queues[3].get()
        if item is None:
            break
        contour_image, image_name = item
        results[image_name] = contour_image
        end_time = time.time()    
    print(f"Total execution time: {end_time - start_time:.6f} seconds")
    print(f"Average time per image: {(end_time - start_time) / len(files):.6f} seconds")
   
    for image_name, contour_image in results.items():
        cv2.imshow(f'Processed Image: {image_name}', contour_image)
        print(f"Showing image: {image_name}. Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()