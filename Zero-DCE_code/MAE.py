from skimage.io import imread
import numpy as np
import os

def calculate_mae(dir1, dir2):
    files1 = sorted([f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))])
    files2 = sorted([f for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))])
    mae_values = []

    for file in files1:
        if file in files2:
            img1 = imread(os.path.join(dir1, file))
            img2 = imread(os.path.join(dir2, file))

            # 确保两个图片尺寸一致
            img1 = np.resize(img1, (min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])))
            img2 = np.resize(img2, (min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])))

            value = np.mean(np.abs(img1 - img2))
            mae_values.append(value)

    average_mae = np.mean(mae_values) if mae_values else 0
    return average_mae

# 示例使用
dir1 = r'/data/ground_truth'
dir2 = r'/data/result/LOL'
average_mae_value = calculate_mae(dir1, dir2)
print(f'Average MAE: {average_mae_value}')