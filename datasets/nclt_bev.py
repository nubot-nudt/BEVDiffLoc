"""
@author: Ziyue Wang and Wen Li
@file: nclt_bev.py
@time: 2025/3/12 14:20
"""

import os
import cv2
import h5py
import torch
import random
import numpy as np
import os.path as osp
from torch.utils import data
from datasets.projection import getBEV
from datasets.augmentor import Augmentor, AugmentParams
from utils.pose_util import process_poses, filter_overflow_nclt, interpolate_pose_nclt, so3_to_euler_nclt, poses_foraugmentaion

BASE_DIR = osp.dirname(osp.abspath(__file__))

velodatatype = np.dtype({
    'x': ('<u2', 0),
    'y': ('<u2', 2),
    'z': ('<u2', 4),
    'i': ('u1', 6),
    'l': ('u1', 7)})

def data2xyzi(data, flip=True):
    xyzil = data.view(velodatatype)
    xyz = np.hstack(
        [xyzil[axis].reshape([-1, 1]) for axis in ['x', 'y', 'z']])
    xyz = xyz * 0.005 - 100.0

    if flip:
        R = np.eye(3)
        R[2, 2] = -1
        xyz = np.matmul(xyz, R)
    return np.hstack([xyz, xyzil['i'].reshape(-1,1)])

def get_velo(velofile):
    return data2xyzi(np.fromfile(velofile))

class NCLT_BEV(data.Dataset):
    def __init__(self, config, split='train'):
        # directories
        if split == 'train':
            self.is_train = True
        else:
            self.is_train = False

        lidar = 'velodyne_left'
        bev = 'merge_bev'
        bev_poses = 'merge_bev.txt'
        data_path = config.train.dataroot

        data_dir = osp.join(data_path, 'NCLT')

        # decide which sequences to use
        if split == 'train':
            split_filename = osp.join(data_dir, 'train_split.txt')
        else:
            split_filename = osp.join(data_dir, 'valid_split.txt')
        with open(split_filename, 'r') as f:
            seqs = [l.rstrip() for l in f if not l.startswith('#')]
        
        ps = {}
        ts = {}
        vo_stats = {}
        self.pcs = []
        self.bev = []
        self.bev_poses = []
        self.merge_num = config.train.merge_num

        for seq in seqs:
            seq_dir = osp.join(data_dir, seq )
            # read the image timestamps
            h5_path = osp.join(seq_dir, lidar + '_' + 'False.h5')
            
            bev_path = osp.join(seq_dir, bev)
            bev_poses_path = osp.join(seq_dir, bev_poses)

            if not os.path.isfile(h5_path):
                print('interpolate ' + seq)
                ts_raw = []
                # 读入LiDAR时间戳，并从小到大排序
                vel = os.listdir(seq_dir + '/velodyne_left')
                for i in range(len(vel)):
                    ts_raw.append(int(vel[i][:-4]))
                ts_raw = sorted(ts_raw)
                # GT poses
                gt_filename = osp.join(seq_dir, 'groundtruth_' + seq + '.csv')
                ts[seq] = filter_overflow_nclt(gt_filename, ts_raw)
                p = interpolate_pose_nclt(gt_filename, ts[seq])  # (n, 6)
                p = so3_to_euler_nclt(p)  # (n, 4, 4)
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

            if self.is_train:
                self.pcs.extend([osp.join(seq_dir, 'velodyne_left', '{:d}.bin'.format(t)) for t in ts[seq]])
                
                merge_sum = 0
                with open(bev_poses_path, 'r') as file:
                    for line in file:
                        bev_pose = list(map(float, line.split()))
                        self.bev_poses.append(bev_pose)
                        merge_sum = merge_sum + 1
                
                for i in range(merge_sum):
                    self.bev.append(osp.join(bev_path, f"{i+1}.png"))
                
            else:
                self.pcs.extend([osp.join(seq_dir, 'velodyne_left', '{:d}.bin'.format(t)) for t in ts[seq]])

        # read / save pose normalization information
        poses = np.empty((0, 12))
        for p in ps.values():
            poses = np.vstack((poses, p))
        pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')
        print("pose_stats_filename:",pose_stats_filename)
        if split == 'train':
            mean_t = np.mean(poses[:, [3, 7, 11]], axis=0)  # (3,)
            std_t = np.std(poses[:, [3, 7, 11]], axis=0)  # (3,)
            np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)

        self.poses = np.empty((0, 6))
        self.rots = np.empty((0, 3, 3))
        for seq in seqs:
            pss, rotation, pss_max, pss_min = process_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t,
                                                            align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
                                                            align_s=vo_stats[seq]['s'])
            self.poses = np.vstack((self.poses, pss))
            self.rots = np.vstack((self.rots, rotation))
            
        # normalize translation
        for bev_pose in self.bev_poses:
            bev_pose[:2] -= mean_t[:2]
            bev_pose[:2] /= std_t[:2]

        if split == 'train':
            print("train data num:" + str(len(self.poses)))
        else:
            print("valid data num:" + str(len(self.poses)))

        augment_params = AugmentParams()
        augment_config = config.augmentation

        # Point cloud augmentations
        if self.is_train:
            augment_params.setTranslationParams(
                p_transx=augment_config['p_transx'], trans_xmin=augment_config[
                    'trans_xmin'], trans_xmax=augment_config['trans_xmax'],
                p_transy=augment_config['p_transy'], trans_ymin=augment_config[
                    'trans_ymin'], trans_ymax=augment_config['trans_ymax'],
                p_transz=augment_config['p_transz'], trans_zmin=augment_config[
                    'trans_zmin'], trans_zmax=augment_config['trans_zmax'])
            augment_params.setRotationParams(
                p_rot_roll=augment_config['p_rot_roll'], rot_rollmin=augment_config[
                    'rot_rollmin'], rot_rollmax=augment_config['rot_rollmax'],
                p_rot_pitch=augment_config['p_rot_pitch'], rot_pitchmin=augment_config[
                    'rot_pitchmin'], rot_pitchmax=augment_config['rot_pitchmax'],
                p_rot_yaw=augment_config['p_rot_yaw'], rot_yawmin=augment_config[
                    'rot_yawmin'], rot_yawmax=augment_config['rot_yawmax'])
            if 'p_scale' in augment_config:
                augment_params.sefScaleParams(
                    p_scale=augment_config['p_scale'],
                    scale_min=augment_config['scale_min'],
                    scale_max=augment_config['scale_max'])
                print(
                    f'Adding scaling augmentation with range [{augment_params.scale_min}, {augment_params.scale_max}] and probability {augment_params.p_scale}')
            self.augmentor = Augmentor(augment_params)
        else:
            self.augmentor = None

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx_N):
        scan_path = self.pcs[idx_N]
        
        pointcloud = get_velo(scan_path)
        
        merge_bev_img = np.empty((0, 3, 251, 251))
        merge_pose = np.empty((0, 3))
        
        if self.is_train:
            random_bev = self.merge_num
            for i in range(random_bev):
                bev_idx = random.randint(0, len(self.bev) - 1)
                bev_path = self.bev[bev_idx]
                bev_pose = np.array(self.bev_poses[bev_idx])
                bev_merge = cv2.imread(bev_path, 0)
                bev_merge = np.tile(bev_merge, (3, 1, 1))
                
                bev_pose = bev_pose.reshape(1, 3)
                bev_merge = np.expand_dims(bev_merge, axis=0)
                
                merge_bev_img = np.concatenate((merge_bev_img, bev_merge), axis=0)
                merge_pose = np.concatenate((merge_pose, bev_pose), axis=0)
        
        if self.is_train:
            pointcloud, rotation = self.augmentor.doAugmentation_bev(pointcloud)  # n, 5
            original_rots = self.rots[idx_N]  # [3, 3]
            present_rots = rotation @ original_rots
            poses = poses_foraugmentaion(present_rots, self.poses[idx_N])
        else:
            poses = self.poses[idx_N]
        
        # Generate BEV_Image
        yaw = poses[5] * 2
        poses_bev = poses[[0, 1]]
        poses_bev = np.hstack((poses_bev, yaw))
        
        pointcloud = pointcloud[:, :3]
        pointcloud = pointcloud[np.where(np.abs(pointcloud[:,0])<50)[0],:]
        pointcloud = pointcloud[np.where(np.abs(pointcloud[:,1])<50)[0],:]
        pointcloud = pointcloud[np.where(np.abs(pointcloud[:,2])<50)[0],:]
        pointcloud = pointcloud.astype(np.float32)
        bev_img = getBEV(pointcloud, 0, 0, 0) # [251, 251]
        bev_img = np.tile(bev_img, (3, 1, 1))
        bev_img_tensor = torch.from_numpy(bev_img.astype(np.float32))
        pose_tensor = torch.from_numpy(poses_bev.astype(np.float32))
        
        merge_bev_img_tensor = torch.from_numpy(merge_bev_img.astype(np.float32))
        merge_pose_tensor = torch.from_numpy(merge_pose.astype(np.float32))

        return bev_img_tensor, pose_tensor, merge_bev_img_tensor, merge_pose_tensor