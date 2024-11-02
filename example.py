import cv2
import numpy as np
import map_code

display = cv2.imread('L2.jpg')
il_map = map_code.get_map(display)
il_map = il_map / np.max(il_map)

image = cv2.imread('10.jpg')

result = np.zeros_like(image, dtype=np.float32)

for i in range(3): 
    result[:, :, i] = image[:, :, i] / il_map

result = cv2.normalize(result, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
cv2.imwrite('cleared.jpg', result)