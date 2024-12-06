import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator, RegularGridInterpolator
from scipy.optimize import curve_fit
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt

def gaussian_2d(coords, A, x0, y0, sigma_x, sigma_y):
    x, y = coords
    return A * np.exp(-((x - x0) ** 2 / (2 * sigma_x ** 2) + (y - y0) ** 2 / (2 * sigma_y ** 2)))

def get_binary_image(image, window_size=128) :
    processed_image = np.zeros_like(image, dtype=np.uint8)
    height, width = image.shape
    for y in range(0, height, window_size):
        for x in range(0, width, window_size):
            window = image[y:y+window_size, x:x+window_size]
            
            mean_intensity = np.mean(window)
            std_intensity = np.std(window)
            
            threshold = mean_intensity + 1.5*std_intensity
            binary_window = (window > threshold)            
            processed_image[y:y+window_size, x:x+window_size][binary_window] = 255
    return processed_image
def get_centroinds(green_channel, i):
    thresh = get_binary_image(green_channel)
    cv2.imwrite(f"MiddleRes/thresh_{i}.jpg", thresh)

    image = cv2.medianBlur(thresh, 15)
    cv2.imwrite(f"MiddleRes/median_{i}.jpg", image)

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = np.array([cv2.contourArea(contour) for contour in contours])
    mean_number = np.mean(areas) - 1.5*np.std(areas)
    filtered_contours = [contour for contour, area in zip(contours, areas) if area >= mean_number]

    centroids = []
    intensities = []

    for contour in filtered_contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            centroids.append((cX, cY))

            mask = np.zeros_like(image)
            cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)
            mean_intensity = cv2.mean(green_channel / 255, mask=mask)[0]
            intensities.append(mean_intensity)

    centroids = np.array(centroids)
    intensities = np.array(intensities)

    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for c in centroids:
        cv2.circle(output_image, (c[0], c[1]), 5, (0, 0, 255), -1)
    cv2.imwrite(f"MiddleRes/centroids_{i}.jpg", output_image)
    return centroids, intensities

def get_map(image, i):
    green_channel = image[:, :, 1]

    centroids, intensities = get_centroinds(green_channel,i)

    rows, cols = image.shape[:2]
    tps = Rbf(centroids[:, 1], centroids[:, 0], intensities, function='thin_plate')

    grid_x, grid_y = np.meshgrid(np.linspace(0, cols-1, cols//3), np.linspace(0, rows-1, rows//3))

    il_map = tps(grid_y, grid_x)
    il_map = cv2.resize(il_map, (cols, rows), interpolation=cv2.INTER_CUBIC)
    return il_map