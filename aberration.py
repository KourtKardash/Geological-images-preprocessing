from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

T1 = 30
T2 = 30

img = io.imread('10.jpg')
img = img.astype(np.int32)

height, width = img.shape[:2]

red_channel = img[:, :, 0]
green_channel = img[:, :, 1]
blue_channel = img[:, :, 2]

D_R = red_channel - green_channel
D_B = blue_channel - green_channel

grad_x_red = ndimage.sobel(red_channel, axis=1)
grad_x_green = ndimage.sobel(green_channel, axis=1)
grad_x_blue = ndimage.sobel(blue_channel, axis=1)


processed_mask = np.zeros(green_channel.shape, dtype=bool)
initial_points = np.argwhere(np.abs(grad_x_green) >= T1)

corrected_img = img.copy()

for p in initial_points:
    px, py = p
    if processed_mask[px, py]:
        continue

    s_p = np.sign(grad_x_green[px, py])

    def H(x, y):
        return max(s_p * grad_x_red[x, y],
               	   s_p * grad_x_green[x, y],
                   s_p * grad_x_blue[x, y])

    m, n = 0, 0

    for i in range(px - 1, -1, -1):
    	if H(i, py) >= T2:
    		m += 1
    	else:
    		break

    for i in range(px + 1, grad_x_green.shape[0]):
        if H(i, py) >= T2:
            n += 1
        else:
            break

    l_p = (max(0, px - m), py)
    r_p = (min(grad_x_green.shape[0] - 1, px + n), py)
    y = py
    for x in range(l_p[0], r_p[0] + 1):
        if D_R[x, y] > max(D_R[l_p], D_R[r_p]):
            corrected_img[x, y, 0] = max(D_R[l_p], D_R[r_p]) + green_channel[x, y]
        if D_R[x, y] < min(D_R[l_p], D_R[r_p]):
            corrected_img[x, y, 0] = min(D_R[l_p], D_R[r_p]) + green_channel[x, y]

        if D_B[x, y] > max(D_B[l_p], D_B[r_p]):
            corrected_img[x, y, 2] = max(D_B[l_p], D_B[r_p]) + green_channel[x, y]
        if D_B[x, y] < min(D_B[l_p], D_B[r_p]):
            corrected_img[x, y, 2] = min(D_B[l_p], D_B[r_p]) + green_channel[x, y]
        processed_mask[x, y] = True
img = np.clip(corrected_img, 0, 255)



red_channel = img[:, :, 0]
green_channel = img[:, :, 1]
blue_channel = img[:, :, 2]

D_R = red_channel - green_channel
D_B = blue_channel - green_channel

grad_y_red = ndimage.sobel(red_channel, axis=0)
grad_y_green = ndimage.sobel(green_channel, axis=0)
grad_y_blue = ndimage.sobel(blue_channel, axis=0)

processed_mask = np.zeros(green_channel.shape, dtype=bool)
initial_points = np.argwhere(np.abs(grad_y_green) >= T1)

corrected_img = img.copy()

for p in initial_points:
    px, py = p
    if processed_mask[px, py]:
        continue

    s_p = np.sign(grad_y_green[px, py])

    def H(x, y):
        return max(s_p * grad_y_red[x, y],
               	   s_p * grad_y_green[x, y],
                   s_p * grad_y_blue[x, y])

    m, n = 0, 0

    for i in range(py - 1, -1, -1):
    	if H(px, i) >= T2:
    		m += 1
    	else:
    		break

    for i in range(py + 1, grad_y_green.shape[1]):
        if H(px, i) >= T2:
            n += 1
        else:
            break

    l_p = (px, max(0, py - m))
    r_p = (px, min(grad_x_green.shape[1] - 1, py + n))
    x = px
    for y in range(l_p[1], r_p[1] + 1):
        if D_R[x, y] > max(D_R[l_p], D_R[r_p]):
            corrected_img[x, y, 0] = max(D_R[l_p], D_R[r_p]) + green_channel[x, y]
        if D_R[x, y] < min(D_R[l_p], D_R[r_p]):
            corrected_img[x, y, 0] = min(D_R[l_p], D_R[r_p]) + green_channel[x, y]

        if D_B[x, y] > max(D_B[l_p], D_B[r_p]):
            corrected_img[x, y, 2] = max(D_B[l_p], D_B[r_p]) + green_channel[x, y]
        if D_B[x, y] < min(D_B[l_p], D_B[r_p]):
            corrected_img[x, y, 2] = min(D_B[l_p], D_B[r_p]) + green_channel[x, y]
        processed_mask[x, y] = True
corrected_img = np.clip(corrected_img, 0, 255)

io.imsave("10_test.jpg", corrected_img.astype(np.uint8))