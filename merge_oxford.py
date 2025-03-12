import os
import h5py
import torch
import cv2
import numpy as np
import os.path as osp
from copy import deepcopy
import open3d as o3d
from torch.utils import data
from datasets.projection import getBEV
from datasets.augmentor import Augmentor, AugmentParams
from utils.pose_util import process_poses, filter_overflow_ts, poses_foraugmentaion
from datasets.robotcar_sdk.python.interpolate_poses import interpolate_ins_poses
from datasets.robotcar_sdk.python.transform import build_se3_transform, euler_to_so3
import time
import math

BASE_DIR = osp.dirname(osp.abspath(__file__))

class Oxford_merge(data.Dataset):
    def __init__(self, split='train'):
        # directories
        if split == 'train':
            self.is_train = True
        else:
            self.is_train = False

        lidar = 'velodyne_left'
        data_path = '/media/wzy/data'

        data_dir = osp.join(data_path, 'Oxford')
        extrinsics_dir = osp.join(BASE_DIR, 'datasets' ,'robotcar_sdk', 'extrinsics')

        seqs = ['2019-01-14-12-05-52']

        ps = {}
        ts = {}
        vo_stats = {}
        self.pcs = []

        # extrinsic reading
        with open(os.path.join(extrinsics_dir, lidar + '.txt')) as extrinsics_file:
            extrinsics = next(extrinsics_file)
        G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])
        with open(os.path.join(extrinsics_dir, 'ins.txt')) as extrinsics_file:
            extrinsics = next(extrinsics_file)
        G_posesource_laser = np.linalg.solve(build_se3_transform([float(x) for x in extrinsics.split(' ')]), G_posesource_laser)  # (4, 4)

        for seq in seqs:
            seq_dir = osp.join(data_dir, seq + '-radar-oxford-10k')
            # read the image timestamps
            h5_path = osp.join(seq_dir, lidar + '_' + 'False.h5')

            if not os.path.isfile(h5_path):
                print('interpolate ' + seq)
                ts_filename = osp.join(seq_dir, lidar + '.timestamps')
                with open(ts_filename, 'r') as f:
                    ts_raw = [int(l.rstrip().split(' ')[0]) for l in f]
                # GT poses
                ins_filename = osp.join(seq_dir, 'gps', 'ins.csv')
                ts[seq] = filter_overflow_ts(ins_filename, ts_raw)
                p = np.asarray(interpolate_ins_poses(ins_filename, deepcopy(ts[seq]), ts[seq][0]))  # (n, 4, 4)
                p = np.asarray([np.dot(pose, G_posesource_laser) for pose in p])  # (n, 4, 4)
                ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))  # (n, 12)
                # write to h5 file
                print('write interpolate pose to ' + h5_path)
                h5_file = h5py.File(h5_path, 'w')
                h5_file.create_dataset('valid_timestamps', data=np.asarray(ts[seq], dtype=np.int64))
                h5_file.create_dataset('poses', data=ps[seq])
            else:
                # load h5 file, save pose interpolating time
                print("load " + seq + ' pose from ' + h5_path)
                h5_file = h5py.File(h5_path, 'r')
                ts[seq] = h5_file['valid_timestamps'][...]
                ps[seq] = h5_file['poses'][...]

            vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

            self.pcs.extend([osp.join(seq_dir, 'velodyne_left', '{:d}.bin'.format(t)) for t in ts[seq]])

        # read / save pose normalization information
        poses = np.empty((0, 12))
        for p in ps.values():
            poses = np.vstack((poses, p))
        pose_stats_filename = osp.join(data_dir, 'Oxford_pose_stats.txt')
        print("pose_stats_filename:",pose_stats_filename)
        if split == 'train':
            mean_t = np.mean(poses[:, [3, 7, 11]], axis=0)  # (3,)
            std_t = np.std(poses[:, [3, 7, 11]], axis=0)  # (3,)
            np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)

        self.poses_3_4 = poses  
        self.poses = np.empty((0, 6))
        self.rots = np.empty((0, 3, 3))
        for seq in seqs:
            pss, rotation, pss_max, pss_min = process_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t,
                                                            align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
                                                            align_s=vo_stats[seq]['s'])
            self.poses = np.vstack((self.poses, pss))
            self.rots = np.vstack((self.rots, rotation))

    def __len__(self):
        return len(self.poses)
    
    def __getitem__(self, idx_N):
        scan_path = self.pcs[idx_N]
        
        pointcloud = np.fromfile(scan_path, dtype=np.float32).reshape(4, -1).transpose()
        pointcloud[:, 2] = -1 * pointcloud[:, 2]
        
        poses_3_4 = self.poses_3_4[idx_N]
        
        # Generate BEV_Image
        pointcloud = pointcloud[:, :3]
        pointcloud = pointcloud[np.where(np.abs(pointcloud[:,0])<50)[0],:]
        pointcloud = pointcloud[np.where(np.abs(pointcloud[:,1])<50)[0],:]
        pointcloud = pointcloud[np.where(np.abs(pointcloud[:,2])<50)[0],:]
        pointcloud = pointcloud.astype(np.float32)
        
        return pointcloud, poses_3_4

if __name__ == '__main__':
    
    dataset = Oxford_merge(split='train')
    merged_pointcloud = o3d.geometry.PointCloud()
    merged_x = []
    merged_y = []
    all_pointcloud = []
    all_poses = []
    voxel_size = 0.4
    image_path = '/home/wzy/merge_bev/'
    pose_path = '/home/wzy/merge_bev.txt'
    with open(pose_path, 'w', encoding='utf-8'):
        pass 
    
    if not os.path.exists(image_path):
    # 如果目录不存在，创建该目录
        os.makedirs(image_path)
        print(f"目录 '{image_path}' 已创建")
    else:
        print(f"目录 '{image_path}' 已存在")
    
    T1 = time.time()
    for i in range(len(dataset)):
        
        pointcloud, poses = dataset[i]
        
        merged_x.append(poses[3])
        merged_y.append(poses[7])
        all_pointcloud.append(pointcloud)
        
        poses = poses.reshape(3, 4)
    
        # 添加最后一行 [0, 0, 0, 1]
        last_row = np.array([0, 0, 0, 1]).reshape(1, 4)
        poses = np.vstack((poses, last_row))
        all_poses.append(poses)
        
        if i > 100:
            
            merged_pointcloud.clear()
            pcd = o3d.geometry.PointCloud()
            
            for j in range(0, len(all_pointcloud), 20):
                # 将pointcloud从numpy数组转为Open3D点云对象
                pcd.points = o3d.utility.Vector3dVector(all_pointcloud[j])
                pcd.transform(all_poses[j])
                merged_pointcloud += pcd
            
            x_mean = np.mean(merged_x)
            y_mean = np.mean(merged_y)
            x_std = np.std(merged_x)
            y_std = np.std(merged_y)
            bev_pointcloud = merged_pointcloud.voxel_down_sample(voxel_size)
            
            # 创建绕Z轴的旋转矩阵
            yaw_random = np.random.uniform(-3.14, 3.14)
            
            x_new = np.random.normal(loc=x_mean, scale=x_std)
            y_new = np.random.normal(loc=y_mean, scale=y_std)
            
            bev_img = getBEV(bev_pointcloud.points, x_new, y_new, yaw_random)
            bev_img = np.tile(bev_img, (3, 1, 1))
            bev_img = bev_img.transpose(1, 2, 0)
            
            cv2.imwrite(f"{image_path}{i-100}.png", bev_img)
            with open(pose_path, 'a') as file:
                file.write(f"{x_new} {y_new} {yaw_random}\n")
            
            all_pointcloud.pop(0)
            all_poses.pop(0)
            merged_x.pop(0)
            merged_y.pop(0)
    
    T2 = time.time()
    print("Time used:", T2-T1)
    
    print("Done")