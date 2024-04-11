from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os

def calculate_ssim(dir1, dir2):
    files1 = sorted([f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))])
    files2 = sorted([f for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))])
    ssim_values = []

    for file in files1:
        if file in files2:
            img1 = imread(os.path.join(dir1, file), as_gray=True)
            img2 = imread(os.path.join(dir2, file), as_gray=True)
            # Resize images to the smallest one's shape
            height = min(img1.shape[0], img2.shape[0])
            width = min(img1.shape[1], img2.shape[1])

            img1_resized = resize(img1, (height, width), anti_aliasing=True)
            img2_resized = resize(img2, (height, width), anti_aliasing=True)

            value = ssim(img1_resized, img2_resized, data_range=img2_resized.max() - img2_resized.min())
            ssim_values.append(value)

    average_ssim = np.mean(ssim_values) if ssim_values else 0
    return average_ssim

# Example usage
dir1 = r'/data/ground_truth'
dir2 = r'/data/result/LOL'
average_ssim = calculate_ssim(dir1, dir2)
print(f'Average SSIM: {average_ssim}')