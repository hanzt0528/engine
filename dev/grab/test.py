import cv2
import numpy as np

# 读取图像
image = cv2.imread('src.png')

# 将图像从BGR颜色空间转换到YCrCb颜色空间
ycc_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

# 提取Y分量（亮度分量）
y_channel = ycc_image[:,:,0]

# 设置阈值，例如设置为200，这表示只保留亮度大于200的像素
# 阈值可以根据图像的具体亮度分布进行调整
threshold = 180

# 创建一个掩膜，只有亮度大于阈值的像素点为255，其余为0
mask = (y_channel > threshold).astype(np.uint8) * 255

# 使用掩膜来保留偏白色的像素
result_image = cv2.bitwise_and(image, image, mask=mask)


cv2.imwrite('dst.png', result_image)
