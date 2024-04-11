from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.io import imread
import numpy as np
import os

def calculate_psnr(dir1, dir2):
    files1 = sorted([f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))])
    files2 = sorted([f for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))])
    psnr_values = []

    for file in files1:
        if file in files2:
            img1 = imread(os.path.join(dir1, file))
            img2 = imread(os.path.join(dir2, file))

            # 确保两个图片尺寸一致
            img1 = np.resize(img1, (min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])))
            img2 = np.resize(img2, (min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])))

            value = psnr(img1, img2)
            psnr_values.append(value)

    average_psnr = np.mean(psnr_values) if psnr_values else 0
    return average_psnr

# 示例使用
dir1 = r'/data/ground_truth'
dir2 = r'/data/result/LOL'
average_psnr_value = calculate_psnr(dir1, dir2)
print(f'Average PSNR: {average_psnr_value}')