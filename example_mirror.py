import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.optimize import minimize

def objective_function(params, image, illumination_map):
    a, b = params
    corrected_image = (image / (a * illumination_map + b))
    ssim_value = -ssim(image, corrected_image, data_range=image.max() - image.min(), win_size=7, channel_axis=-1)
    regularization = (a - 1) ** 2 
    return ssim_value + regularization


image = cv2.imread("39.jpg")

mirror = cv2.imread('mirror3200.jpg')
mirror = cv2.cvtColor(mirror, cv2.COLOR_BGR2GRAY)
mirror = mirror / 255
mirror = mirror/np.max(mirror)
mask_3ch = np.repeat(mirror[:, :, np.newaxis], 3, axis=2)
initial_params = [1, 0.5]
bounds = [(0.1, 2), (0.01, 1)]
result = minimize(objective_function, initial_params, args=(image, mask_3ch), bounds=bounds)
a_opt, b_opt = result.x
corrected_im = image / (a_opt * mask_3ch + b_opt)
cv2.imwrite(f"39_corr.jpg", corrected_im)
