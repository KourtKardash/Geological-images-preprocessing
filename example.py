import cv2
import numpy as np
import map_code
import matplotlib.pyplot as plt

image = cv2.imread("39.jpg")
display = cv2.imread("L14.jpg")
il_map = map_code.get_map(display)

max_intens = np.max(il_map)
delta = 1 - max_intens
il_map1 = il_map + delta

mask_3ch = np.repeat(il_map1[:, :, np.newaxis], 3, axis=2)
corrected_image = image / mask_3ch
#res = cv2.normalize(corrected_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
cv2.imwrite("39_out.jpg", np.uint8(corrected_image))