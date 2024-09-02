import numpy as np

def rotate_point_2d(point, angle):
    """
    Rotate a point in 2D space around the origin by a given angle in degrees.
    
    Parameters:
    - point: A tuple or list representing the point (x, y).
    - angle: The rotation angle in degrees.
    
    Returns:
    - A tuple representing the rotated point.
    """
    # 将角度转换为弧度
    angle_rad = np.deg2rad(angle)
    
    # 创建旋转矩阵
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    # 将点转换为NumPy数组
    point_array = np.array(point)
    
    # 应用旋转矩阵
    rotated_point_array = np.dot(rotation_matrix, point_array)
    
    # 将结果转换回元组形式
    rotated_point = tuple(rotated_point_array)
    
    return rotated_point

# 示例：旋转点 (1, 0) 90 度
original_point = (1,0)
angle = 90  # 旋转角度
#org (1,0)
#(2,-1) ->(1,2)

#org (2,0)
#(2,-2) ->(2,2)

#org(0,-2)

#(0,0)

#org(0,-1)
#(1,0) ->(0,1)

original_point = (1,0)
angle = 90  # 旋转角度
rotated_point = rotate_point_2d(original_point, angle)
print(f"Original point: {original_point}")
print(f"Rotated point: {rotated_point}")