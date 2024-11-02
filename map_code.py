import cv2
import numpy as np
from scipy.optimize import curve_fit

N = 256

def local_deviation_thresholding(image, window_size=64, threshold_factor=1.5):
    binary_image = np.zeros_like(image, dtype=np.uint8)
    height, width = image.shape

    # Проходимся по каждому окну изображения
    for y in range(0, height, window_size):
        for x in range(0, width, window_size):
            # Извлекаем текущее окно
            window = image[y:y+window_size, x:x+window_size]
            
            # Рассчитываем среднее и стандартное отклонение для текущего окна
            mean_intensity = np.mean(window)
            std_intensity = np.std(window)
            
            # Определяем порог как среднее + коэффициент * стандартное отклонение
            threshold = mean_intensity + threshold_factor * std_intensity
            
            # Применяем порог для создания бинарного окна
            binary_window = (window > threshold)
            
            # Сохраняем бинаризованное окно в итоговое изображение
            binary_image[y:y+window_size, x:x+window_size][binary_window] = window[binary_window]

    return binary_image

def gaussian_2d(coords, A, x0, y0, sigma_x, sigma_y):
    x, y = coords
    return A * np.exp(-((x - x0) ** 2 / (2 * sigma_x ** 2) + (y - y0) ** 2 / (2 * sigma_y ** 2)))

def get_centroinds(green_channel):
    processed_image = local_deviation_thresholding(green_channel)

    cv2.imwrite("test_local.jpg", processed_image)

    image = cv2.medianBlur(processed_image, 7) 

    cv2.imwrite("test_median.jpg", image)

    blurred_image = cv2.GaussianBlur(image, (13, 13), 0)

    cv2.imwrite("test_blurred.jpg", blurred_image)

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
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for c in centroids:
        cv2.circle(output_image, (c[0], c[1]), 5, (0, 255, 0), -1)
    cv2.imwrite("test_centr.jpg", output_image)
    return centroids, intensities

def get_map():
    image = cv2.imread('L1.jpg')
    green_channel = image[:, :, 1]

    centroids, intensities = get_centroinds(green_channel)
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
Z = get_map()
cv2.imwrite("test_map.jpg", (Z*255).astype(np.uint8))