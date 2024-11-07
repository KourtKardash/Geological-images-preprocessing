import cv2
import numpy as np
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def gaussian_2d(coords, A, x0, y0, sigma_x, sigma_y):
    x, y = coords
    return A * np.exp(-((x - x0) ** 2 / (2 * sigma_x ** 2) + (y - y0) ** 2 / (2 * sigma_y ** 2)))

def get_binary_image(image, window_size=64) :
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
    blurred_image = cv2.GaussianBlur(green_channel, (15, 15), 0)
    cv2.imwrite(f"MiddleRes/blurred_{i}.jpg", blurred_image)

    thresh = get_binary_image(blurred_image)
    cv2.imwrite(f"MiddleRes/thresh_{i}.jpg", thresh)

    image = cv2.medianBlur(thresh, 15) 
    cv2.imwrite(f"MiddleRes/median_{i}.jpg", image)

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = np.array([cv2.contourArea(contour) for contour in contours])
    mean_number = np.mean(areas) - np.std(areas)
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
    #print("start pca")
    #pca = PCA(n_components=2)
    #pca.fit(centroids)
    #angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
    #print(angle)
    #print("end pca")
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for c in centroids:
        cv2.circle(output_image, (c[0], c[1]), 5, (0, 0, 255), -1)
    cv2.imwrite(f"MiddleRes/centroids_{i}.jpg", output_image)
    return centroids, intensities

def get_map(image, i):
    green_channel = image[:, :, 1]

    centroids, intensities = get_centroinds(green_channel,i)
    x_data = centroids[:, 0]
    y_data = centroids[:, 1]

    initial_guess = (1, np.mean(x_data), np.mean(y_data), 1000, 1000)

    popt, pcov = curve_fit(gaussian_2d, (x_data, y_data), intensities, p0=initial_guess)

    A_opt, x0_opt, y0_opt, sigma_x_opt, sigma_y_opt = popt

    height, width = image.shape[:2]
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    X, Y = np.meshgrid(x, y)

    Z = gaussian_2d((X, Y), A_opt, x0_opt, y0_opt, sigma_x_opt, sigma_y_opt)
    #np.save('mask.npy', Z)
    return Z