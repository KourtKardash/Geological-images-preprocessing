import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("04.jpg")
image = image / 255
mirror = cv2.imread('mirror1200.jpg')
mirror = cv2.cvtColor(mirror, cv2.COLOR_BGR2GRAY)
mirror = mirror / 255
mirror = cv2.GaussianBlur(mirror, (51, 51), 0)
delta = 1 - np.max(mirror)

mirror1 = mirror + delta

mask_3ch = np.repeat(mirror1[:, :, np.newaxis], 3, axis=2)
corrected_im = image / mask_3ch
corrected_im = np.clip(corrected_im, 0, 1)

cv2.imwrite(f"04_out_mirror.jpg", np.uint8(corrected_im * 255))
