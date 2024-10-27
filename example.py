import cv2
import numpy as np
import map_code

il_map = map_code.get_map()
image = cv2.imread('S1_v1/imgs/test/02.jpg')

result = np.zeros_like(image, dtype=np.float32)

for i in range(3): 
    result[:, :, i] = image[:, :, i] / il_map

result = cv2.normalize(result, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
cv2.imwrite('cleared.jpg', result)