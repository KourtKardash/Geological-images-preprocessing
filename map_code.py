import cv2
import numpy as np
from scipy.optimize import curve_fit

N = 256
image = cv2.imread('L1.jpg')
red_channel = image[:, :, 0]
LEFT_SHIFT = 10
RIGHT_SHIFT = 33

def process_local_region(region):
    hist, bin_edges = np.histogram(region.flatten(), bins=N, range=[0, N])

    peak_value = np.max(hist)
    peak_index = np.argmax(hist)

    left_bound = peak_index + LEFT_SHIFT
    right_bound = min(N, peak_index + RIGHT_SHIFT)

    mask = (region >= left_bound) & (region <= right_bound)

    filtered_region = np.zeros_like(region)
    filtered_region[mask] = region[mask]
    
    return filtered_region

def process_image_locally(image, region_size, N=256):
    height, width = image.shape
    processed_image = np.zeros_like(image)

    for y in range(0, height, region_size):
        for x in range(0, width, region_size):
            y_end = min(y + region_size, height)
            x_end = min(x + region_size, width)

            region = image[y:y_end, x:x_end]
            filtered_region = process_local_region(region)
            processed_image[y:y_end, x:x_end] = filtered_region

    return processed_image

def gaussian_2d(coords, A, x0, y0, sigma_x, sigma_y):
    x, y = coords
    return A * np.exp(-((x - x0) ** 2 / (2 * sigma_x ** 2) + (y - y0) ** 2 / (2 * sigma_y ** 2)))

def get_centroinds():
    processed_image = process_image_locally(red_channel, region_size=64)
    image = cv2.medianBlur(processed_image, 7) 

    blurred_image = cv2.GaussianBlur(image, (13, 13), 0)
    _, thresh = cv2.threshold(blurred_image, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    intensities = []

    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            centroids.append((cX, cY))

            mask = np.zeros_like(image)
            cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)
            mean_intensity = cv2.mean(image / 255, mask=mask)[0]
            intensities.append(mean_intensity)

    centroids = np.array(centroids)
    intensities = np.array(intensities)
    #output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    #for c in centroids:
    #    cv2.circle(output_image, (c[0], c[1]), 5, (0, 255, 0), -1)
    #cv2.imwrite("centroids_clear.jpg", output_image)
    return centroids, intensities

def get_map():
    centroids, intensities = get_centroinds()
    x_data = centroids[:, 0]
    y_data = centroids[:, 1]

    initial_guess = (1, np.mean(x_data), np.mean(y_data), 10, 10)

    popt, pcov = curve_fit(gaussian_2d, (x_data, y_data), intensities, p0=initial_guess)
    A_opt, x0_opt, y0_opt, sigma_x_opt, sigma_y_opt = popt

    height, width = image.shape[:2]
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    X, Y = np.meshgrid(x, y)

    Z = gaussian_2d((X, Y), A_opt, x0_opt, y0_opt, sigma_x_opt, sigma_y_opt)
    #np.save('mask.npy', Z)
    return Z