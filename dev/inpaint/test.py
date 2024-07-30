import cv2
import numpy as np

# 读取图像和掩膜（损坏区域）
image = cv2.imread('src.png')
mask = cv2.imread('mask.png', 0)

# 使用inpaint函数修复图像
dst = cv2.inpaint(image, mask, 9, cv2.INPAINT_NS)

# 显示和保存修复后的图像
cv2.imwrite("dst.png",dst)
