import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_centroids(file_path):
    centroids = np.loadtxt(file_path, delimiter=',')
    return centroids


def match_centroids_to_grid_matrix(x_coords_rotated, y_coords_rotated, centroids, row, col, height, width, min_distance):
    matched = {}
    unused_centroids = centroids.tolist()

    for i in range (row, -1, -1) :
        delta_x, delta_y = 0, 0 
        for j in range (col, x_coords_rotated.shape[1]) :
            x = x_coords_rotated[i, j]
            y = y_coords_rotated[i, j]
            shifted_x, shifted_y = x + delta_x, y + delta_y
            distances = np.sqrt((centroids[:, 0] - shifted_x) ** 2 + (centroids[:, 1] - shifted_y) ** 2)
            nearest_idx = np.argmin(distances)
            nearest_centroid = centroids[nearest_idx]
            if any(np.array_equal(nearest_centroid, centroid) for centroid in unused_centroids) and np.min(distances) < min_distance:
                matched[(x, y)] = tuple(nearest_centroid)
                for idx, centroid in enumerate(unused_centroids):
                    if np.array_equal(centroid, nearest_centroid):
                        del unused_centroids[idx]
                        break
                delta_x = nearest_centroid[0] - x
                delta_y = nearest_centroid[1] - y

    for i in range (row + 1, x_coords_rotated.shape[0]) :
        delta_x, delta_y = 0, 0 
        for j in range (col, x_coords_rotated.shape[1]) :
            x = x_coords_rotated[i, j]
            y = y_coords_rotated[i, j]
            shifted_x, shifted_y = x + delta_x, y + delta_y
            distances = np.sqrt((centroids[:, 0] - shifted_x) ** 2 + (centroids[:, 1] - shifted_y) ** 2)
            nearest_idx = np.argmin(distances)
            nearest_centroid = centroids[nearest_idx]
            if any(np.array_equal(nearest_centroid, centroid) for centroid in unused_centroids) and np.min(distances) < min_distance:
                matched[(x, y)] = tuple(nearest_centroid)
                for idx, centroid in enumerate(unused_centroids):
                    if np.array_equal(centroid, nearest_centroid):
                        del unused_centroids[idx]
                        break
                delta_x = nearest_centroid[0] - x
                delta_y = nearest_centroid[1] - y

    for i in range (row + 1, x_coords_rotated.shape[0]) :
        delta_x, delta_y = 0, 0 
        for j in range (col - 1, -1, -1) :
            x = x_coords_rotated[i, j]
            y = y_coords_rotated[i, j]
            shifted_x, shifted_y = x + delta_x, y + delta_y
            distances = np.sqrt((centroids[:, 0] - shifted_x) ** 2 + (centroids[:, 1] - shifted_y) ** 2)
            nearest_idx = np.argmin(distances)
            nearest_centroid = centroids[nearest_idx]
            if any(np.array_equal(nearest_centroid, centroid) for centroid in unused_centroids) and np.min(distances) < min_distance:
                matched[(x, y)] = tuple(nearest_centroid)
                for idx, centroid in enumerate(unused_centroids):
                    if np.array_equal(centroid, nearest_centroid):
                        del unused_centroids[idx]
                        break
                delta_x = nearest_centroid[0] - x
                delta_y = nearest_centroid[1] - y

    for i in range (row, -1, -1) :
        delta_x, delta_y = 0, 0 
        for j in range (col - 1, -1, -1) :
            x = x_coords_rotated[i, j]
            y = y_coords_rotated[i, j]
            shifted_x, shifted_y = x + delta_x, y + delta_y
            distances = np.sqrt((centroids[:, 0] - shifted_x) ** 2 + (centroids[:, 1] - shifted_y) ** 2)
            nearest_idx = np.argmin(distances)
            nearest_centroid = centroids[nearest_idx]
            if any(np.array_equal(nearest_centroid, centroid) for centroid in unused_centroids) and np.min(distances) < min_distance:
                matched[(x, y)] = tuple(nearest_centroid)
                for idx, centroid in enumerate(unused_centroids):
                    if np.array_equal(centroid, nearest_centroid):
                        del unused_centroids[idx]
                        break
                delta_x = nearest_centroid[0] - x
                delta_y = nearest_centroid[1] - y

        # Вывод результата
    #for (x, y), centroid in matched.items():
    #    print(f"Узел ({x}, {y}) -> Центроид {centroid}")
    return matched


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


def rotate_grid(x_coords, y_coords, central_centroid, tg_angle):
    theta = np.arctan(tg_angle)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    translated_points = np.vstack([x_coords.ravel() - central_centroid[0], y_coords.ravel() - central_centroid[1]])
    rotated_points = rotation_matrix @ translated_points

    rotated_points[0] += central_centroid[0]
    rotated_points[1] += central_centroid[1]

    x_coords_rotated = rotated_points[0].reshape(x_coords.shape)
    y_coords_rotated = rotated_points[1].reshape(y_coords.shape)

    return x_coords_rotated, y_coords_rotated


def get_centroinds(green_channel, ind):
    thresh = get_binary_image(green_channel)
    image = cv2.medianBlur(thresh, 9)
    cv2.imwrite(f'MiddleRes/median_{ind}.png', image)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = np.array([cv2.contourArea(contour) for contour in contours])
    mean_number = np.mean(areas) - 1.5*np.std(areas)
    filtered_contours = [contour for contour, area in zip(contours, areas) if area >= mean_number]

    centroids = []

    for contour in filtered_contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            centroids.append((cX, cY))

    centroids = np.array(centroids)

    height, width = image.shape

    center_x_min = width * 0.4
    center_x_max = width * 0.6
    center_y_min = height * 0.49
    center_y_max = height * 0.51

    central_centroids = []
    while len(central_centroids) < 2 :
        central_centroids = [
            (x, y) for x, y in centroids
            if center_x_min <= x <= center_x_max and center_y_min <= y <= center_y_max
        ]
        center_y_max = center_y_max + 10
        center_y_min = center_y_min + 10
    central_centroids = np.array(central_centroids)

    min_distance = float('inf')
    closest_pair = None

    for i in range(len(central_centroids)):
        for j in range(i + 1, len(central_centroids)):
            distance = np.linalg.norm(central_centroids[i] - central_centroids[j])
            if distance < min_distance:
                min_distance = distance
                closest_pair = (central_centroids[i], central_centroids[j])
    
    central_centroids = sorted(central_centroids, key=lambda point: point[0])
    central_centroid = central_centroids[0]
    end_centroid = central_centroids[-1]

    x_coords_right = np.arange(central_centroid[0]+ min_distance, width, min_distance)
    x_coords_left = np.arange(central_centroid[0], 0, -min_distance)
    x_coords_left = x_coords_left[::-1]
    x_coords = np.concatenate((x_coords_left, x_coords_right))
    index_x = len(x_coords_left)

    y_coords_up = np.arange(central_centroid[1], 0, -min_distance)
    y_coords_up = y_coords_up[::-1]
    y_coords_down = np.arange(central_centroid[1] + min_distance, height, min_distance)
    y_coords = np.concatenate((y_coords_up, y_coords_down))
    index_y = len(y_coords_up)

    x_coords, y_coords = np.meshgrid(x_coords, y_coords)
    t_angle = (end_centroid[1] - central_centroid[1]) / (end_centroid[0] - central_centroid[0])
    x_coords_rotated , y_coords_rotated= rotate_grid(x_coords, y_coords, central_centroid, t_angle)

    start_row, start_col = index_y, index_x

    output_image = np.zeros_like(image)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
    for c in centroids:
        if any((c == point).all() for point in central_centroids): 
            cv2.circle(output_image, (c[0], c[1]), 5, (0,0,255), -1)
        else:
            cv2.circle(output_image, (c[0], c[1]), 5, (255,0,0), -1)
    cv2.imwrite(f'MiddleRes/centroids_{ind}.png', output_image)

    return match_centroids_to_grid_matrix(x_coords_rotated, y_coords_rotated, centroids, start_row-1, start_col-1, height, width, min_distance)


def draw_grid(image, x_coords, y_coords, color=(0, 255, 0), thickness=1):
    for y in range(y_coords.shape[0]):
        cv2.line(image, (int(x_coords[y, 0]), int(y_coords[y, 0])), (int(x_coords[y, -1]), int(y_coords[y, -1])), color, thickness)
    for x in range(x_coords.shape[1]):
        cv2.line(image, (int(x_coords[0, x]), int(y_coords[0, x])), (int(x_coords[-1, x]), int(y_coords[-1, x])), color, thickness)
    return image





image = cv2.imread('L5.jpg', cv2.IMREAD_COLOR)
grid_to_centroid = get_centroinds(image[:, :, 1], 5)

#centroid_image = cv2.imread('MiddleRes/centroids_5.png')
#grid_image = draw_grid(centroid_image, x, y)

#cv2.imwrite('grid_image_on_centroids_5.png', grid_image)

object_points = np.array([(x, y, 0) for (x, y) in grid_to_centroid.keys()], dtype=np.float32)
image_points = np.array([value for value in grid_to_centroid.values()], dtype=np.float32)

image_size = image.shape[:2]
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    [object_points],
    [image_points],
    image_size, None, None
)
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix,
    dist_coeffs,
    image_size,
    0,
    image_size
)

# Загружаем изображение
for i in range (1,6):
    image = cv2.imread(f"kedr3/4.{i}.jpg")

    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

    cv2.imwrite(f"kedr3_undistorced/4.{i}.jpg", undistorted_image)