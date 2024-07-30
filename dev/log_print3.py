import matplotlib.pyplot as plt

# 假设我们有一些数据
tasks = ['Task 1', 'Task 2', 'Task 3', 'Task 4']
cpu_times = [0.7, 3.4, 1.8, 2.9]
gpu_times = [0.7, 1.1, 0.6, 0.9]

# 创建条形图
fig, ax = plt.subplots()
cpu_bar = ax.bar(tasks, cpu_times, label='CPU Time', color='blue')
gpu_bar = ax.bar(tasks, gpu_times, label='GPU Time', color='green', bottom=cpu_times)

# 为每个条形添加文本标签
def add_text_labels(bar_container, value_func, text_fmt='{}'):
    for bar in bar_container.patches:
        height = value_func(bar.get_height())
        ax.text(bar.get_x() + bar.get_width()/2, height/2, text_fmt.format(height),
                ha='center', va='bottom')

# 添加 CPU 时间的文本标签
add_text_labels(cpu_bar, lambda h: h, text_fmt='CPU: {:.1f}')

# 添加 GPU 时间的文本标签
add_text_labels(gpu_bar, lambda h: sum(cpu_times) - h, text_fmt='GPU: {:.1f}')

# 添加图例
plt.savefig('log.png')