import matplotlib.pyplot as plt
 
# 准备数据
labels = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
values = [4.87, 9.39, 11.36, 20.83, 60.34]
 
# 画柱状图
fig, ax = plt.subplots()
bars = plt.bar(labels, values)
 
 # 在每个柱子上显示数值
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 在文本上方3点的位置显示
                textcoords="offset points",
                ha='center', va='bottom')
    
# ax.set_title('柱状图示例')
# ax.set_xlabel('类别')

ax.set_ylabel('seconds')

# 显示图形
plt.savefig("performance.png")
plt.show()