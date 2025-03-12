"""
@author: Ziyue Wang and Lun Lou
@file: projection.py
@time: 2025/3/12 14:20
"""

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
import os
import argparse
from tqdm import trange
import math

def getBEV(all_points, midx, midy, yaw): #N*3
    
    all_points_pc = o3d.geometry.PointCloud()# pcl.PointCloud()
    all_points_pc.points = o3d.utility.Vector3dVector(all_points)#all_points_pc.from_array(all_points)
    all_points_pc = all_points_pc.voxel_down_sample(voxel_size=0.4) #f = all_points_pc.make_voxel_grid_filter()
    
    # 定义平移向量（例如平移 [1, 2, 3]）
    translation = np.array([-midx, -midy, 0])

    # 创建平移矩阵（4x4），对角线为1，最后一列是平移向量
    transformation_matrix = np.eye(4)  # 生成一个4x4单位矩阵
    transformation_matrix[:3, 3] = translation  # 只修改最后一列，保持旋转部分为单位矩阵
    all_points_pc.transform(transformation_matrix)
    
    rotation = np.array([
        [math.cos(-yaw), -math.sin(-yaw), 0],
        [math.sin(-yaw),  math.cos(-yaw), 0],
        [0, 0, 1]
    ])
    # 创建旋转矩阵（4x4）
    rotation_matrix = np.eye(4)  # 生成一个4x4单位矩阵
    rotation_matrix[:3, :3] = rotation  # 只修改最后一列，保持旋转部分为单位矩阵
    all_points_pc.transform(rotation_matrix)
    
    all_points = np.asarray(all_points_pc.points)# np.array(all_points_pc.to_list())

    x_min = -50
    y_min = -50
    x_max = +50 
    y_max = +50

    x_min_ind = np.floor(x_min/0.4).astype(int)
    x_max_ind = np.floor(x_max/0.4).astype(int)
    y_min_ind = np.floor(y_min/0.4).astype(int)
    y_max_ind = np.floor(y_max/0.4).astype(int)

    x_num = x_max_ind - x_min_ind + 1
    y_num = y_max_ind - y_min_ind + 1

    mat_global_image = np.zeros((y_num,x_num),dtype=np.uint8)
    
    for i in range(all_points.shape[0]):
        x_ind = x_max_ind-np.floor((all_points[i,1])/0.4).astype(int)
        y_ind = y_max_ind-np.floor((all_points[i,0])/0.4).astype(int)
        if(x_ind >= x_num or y_ind >= y_num or x_ind < 0 or y_ind < 0):
            continue
        if mat_global_image[ y_ind,x_ind]<10:
            mat_global_image[ y_ind,x_ind] += 1

    max_pixel = np.max(np.max(mat_global_image))

    mat_global_image[mat_global_image<=1] = 0  
    mat_global_image = mat_global_image*10
    
    mat_global_image[np.where(mat_global_image>255)]=255
    mat_global_image = mat_global_image/np.max(mat_global_image)*255

    return mat_global_image