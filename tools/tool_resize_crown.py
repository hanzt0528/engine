from PIL import Image
import os

# 设置目标目录和原始目录
target_directory = '/mnt/cephfs/data/projects/博兴轮胎/20240508/小胎/crown/'
source_directory = '/mnt/cephfs/data/projects/博兴轮胎/20240508/小胎/胎冠PNG'

# 遍历原始目录中的所有文件
for filename in os.listdir(source_directory):
    if filename.endswith('.png'):
        # 构造完整的文件路径
        source_path = os.path.join(source_directory, filename)
        
        # 打开图片
        with Image.open(source_path) as img:
            # 调整图片大小到指定的分辨率
            img_resized = img.resize((2048, 5000), Image.ANTIALIAS)
            
            # 构造目标路径
            target_path = os.path.join(target_directory, filename)
            
            # 保存调整大小后的图片
            img_resized.save(target_path, 'PNG')
        

print("图片尺寸调整完成。")