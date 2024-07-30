import matplotlib.pyplot as plt
import numpy as np

# 假设我们有若干任务的 CPU 和 GPU 执行时间
tasks = ['conv2d', 'Task 2', 'Task 3', 'Task 4', 'Task 5']
cpu_times = [1749, 2.9, 1.5, 3.4, 2.7]  # CPU 执行时间（秒）
gpu_times = [111, 0.7, 0.5, 0.9, 0.6]  # GPU 执行时间（秒）

# 设置条形图的宽度
bar_width = 0.35

# 绘制条形图
fig, ax = plt.subplots()

# 为 CPU 和 GPU 的条形图设置位置
cpu_bar = ax.bar(tasks, cpu_times, bar_width, label='CPU Time', color='blue')
gpu_bar = ax.bar([t + bar_width for t in range(len(tasks))], gpu_times, bar_width, label='GPU Time', color='green')
# 添加图例
plt.legend()

# 添加类别标签
#plt.xticks([t + bar_width / 2 for t in tasks], tasks)

# 添加标题和轴标签
plt.title('CPU vs. GPU Execution Time')
plt.xlabel('Tasks')
plt.ylabel('Execution Time (microseconds)')

# 显示数值标签
def add_values(ax, spacing=5):
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        va = 'bottom' if y_value < 0 else 'top'
        ax.text(x_value, y_value + spacing, round(y_value, 2), ha='center', va=va)
        

# # 添加数值标签到条形图中
add_values(cpu_bar)
add_values(gpu_bar)

# 显示图形
plt.savefig('log.png')