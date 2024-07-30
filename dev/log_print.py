import matplotlib.pyplot as plt
import numpy as np

# 打开文件

names = []
values = []
with open('1.txt', 'r', encoding='utf-8') as file:
    # 逐行读取文件内容
    for line in file:
        # 根据需要解析每一行
        # 例如，这里简单地打印每一行
        line = line.strip()
        parts = line.split(':')
        names.append(parts[0])
        values.append(float(parts[1]))

print(names)
print(values)

# 创建一个随机数组
data = np.random.randn(1000)

# 创建图形
plt.figure()

# 绘制直方图
plt.hist(data, bins=30, alpha=0.5, color='r')

# 添加标题
plt.title('Histogram')

# 显示图形
plt.savefig('log.png')
