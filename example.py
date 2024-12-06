import cv2
import numpy as np
import map_code
from skimage.metrics import structural_similarity as ssim
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def objective_function(params, image, illumination_map):
    a, b = params
    corrected_image = (image / (a * illumination_map + b))
    ssim_value = -ssim(image, corrected_image, data_range=image.max() - image.min(), win_size=7, channel_axis=-1)
    regularization = (a - 1) ** 2 
    return ssim_value + regularization

image = cv2.imread("39.jpg")
il_map = map_code.get_map(image)
mask_3ch = np.repeat(il_map[:, :, np.newaxis], 3, axis=2)
initial_params = [1, 0.5]
bounds = [(0.1, 2), (0, 1)]
result = minimize(objective_function, initial_params, args=(image, mask_3ch), bounds=bounds)
a_opt, b_opt = result.x
corrected_image = image / (a_opt * mask_3ch + b_opt)
res = cv2.normalize(corrected_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
cv2.imwrite("39_out.jpg", res)
