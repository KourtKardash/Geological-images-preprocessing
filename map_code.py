import cv2
import numpy as np
from scipy.optimize import curve_fit

def gaussian_2d(coords, A, x0, y0, sigma_x, sigma_y):
    x, y = coords
    return A * np.exp(-((x - x0) ** 2 / (2 * sigma_x ** 2) + (y - y0) ** 2 / (2 * sigma_y ** 2)))

def get_binary_image(chosen_channel, add_channel_1, add_channel_2, window_size=128) :
    processed_image = np.zeros_like(chosen_channel, dtype=np.uint8)
    height, width = chosen_channel.shape
    for y in range(0, height, window_size):
        for x in range(0, width, window_size):
            window = chosen_channel[y:y+window_size, x:x+window_size]
            add_window_1 = add_channel_1[y:y+window_size, x:x+window_size]
            add_window_2 = add_channel_2[y:y+window_size, x:x+window_size]

            mean_intensity = np.mean(window)
            std_intensity = np.std(window)
            
            threshold = mean_intensity + std_intensity

            binary_window = ((window > threshold) & (add_window_1 < np.mean(add_window_1) + \
                2*np.std(add_window_1)) & (add_window_2 < np.mean(add_window_2) + 2*np.std(add_window_2)))
            processed_image[y:y+window_size, x:x+window_size][binary_window] = 255
    return processed_image

def get_centroinds(chosen_channel, add_channel_1, add_channel_2):
    thresh = get_binary_image(chosen_channel, add_channel_1, add_channel_2)
    image = cv2.medianBlur(thresh, 9)

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = np.array([cv2.contourArea(contour) for contour in contours])
    mean_number = np.mean(areas) - 1.5 * np.std(areas)
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
            mean_intensity = cv2.mean(chosen_channel / 255, mask=mask)[0]
            intensities.append(mean_intensity)

    centroids = np.array(centroids)
    intensities = np.array(intensities)

    #np.savetxt('centroids1.txt', centroids, delimiter=',', fmt='%d')
    
    '''
    output_image = image
    output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
    for c in centroids:
        cv2.circle(output_image, (c[0], c[1]), 7, (0, 0, 255), -1)
    cv2.imwrite(f'MiddleRes/centroids.png', output_image)
    '''
    return centroids, intensities

def get_map(image):
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    height, width = red_channel.shape

    x_min, x_max = int(width * 0.4), int(width * 0.6)
    y_min, y_max = int(height * 0.4), int(height * 0.6)
    min_intensities = [
        np.min(channel[x_min:x_max, y_min:y_max])
        for channel in [red_channel, green_channel, blue_channel]
    ]
    min_index = np.argmin(min_intensities)
    chosen_channel = [red_channel, green_channel, blue_channel][min_index]
    add_channel_1 = [green_channel, blue_channel, red_channel][min_index]
    add_channel_2 = [blue_channel, red_channel, green_channel][min_index]

    centroids, intensities = get_centroinds(chosen_channel, add_channel_1, add_channel_2)
    x_data = centroids[:, 0]
    y_data = centroids[:, 1]

    initial_guess = (1, np.mean(x_data), np.mean(y_data), 1000, 1000)

    popt, pcov = curve_fit(gaussian_2d, (x_data, y_data), intensities, p0=initial_guess, maxfev=10000)

    A_opt, x0_opt, y0_opt, sigma_x_opt, sigma_y_opt = popt

    height, width = image.shape[:2]
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    X, Y = np.meshgrid(x, y)

    Z = gaussian_2d((X, Y), A_opt, x0_opt, y0_opt, sigma_x_opt, sigma_y_opt)
    
    #np.save('mask.npy', Z)
    return Z

