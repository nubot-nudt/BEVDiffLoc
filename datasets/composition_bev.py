"""
@author: Ziyue Wang and Wen Li
@file: composition_bev.py
@time: 2025/3/12 14:20
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '../')

from .oxford_bev import Oxford_BEV
from .nclt_bev import NCLT_BEV
from torch.utils import data
from utils.pose_util import calc_vos_safe_fc


class MF_bev(data.Dataset):
    def __init__(self, dataset, config, split='train', include_vos=False):

        self.steps = config.train.steps
        self.skip = config.train.skip
        self.use_merge = config.train.use_merge
        self.train = split

        if dataset == 'Oxford':
            self.dset = Oxford_BEV(config, split)
        elif dataset == 'NCLT':
            self.dset = NCLT_BEV(config, split)
        else:
            raise NotImplementedError('{:s} dataset is not implemented!')

        self.L = self.steps * self.skip
        # GCS
        self.include_vos = include_vos
        self.vo_func = calc_vos_safe_fc


    def get_indices(self, index):
        skips = self.skip * np.ones(self.steps-1)
        offsets = np.insert(skips, 0, 0).cumsum()  # (self.steps,)
        offsets -= offsets[len(offsets) // 2]
        offsets = offsets.astype(np.int_)
        idx = index + offsets
        idx = np.minimum(np.maximum(idx, 0), len(self.dset)-1)
        assert np.all(idx >= 0), '{:d}'.format(index)
        assert np.all(idx < len(self.dset))
        return idx
    
    def get_merge_indices(self, index):
        skips = self.merge_skip * np.ones(self.merge_steps-1)
        offsets = np.insert(skips, 0, 0).cumsum()  # (self.steps,)
        offsets -= offsets[len(offsets) // 2]
        offsets = offsets.astype(np.int_)
        idx = index + offsets
        idx = np.minimum(np.maximum(idx, 0), len(self.dset)-1)
        assert np.all(idx >= 0), '{:d}'.format(index)
        assert np.all(idx < len(self.dset))
        return idx

    def __getitem__(self, index):
        idx         = self.get_indices(index)

        clip        = [self.dset[i] for i in idx]
        pcs         = torch.stack([c[0] for c in clip], dim=0)  # (self.steps, 1, 251, 251) 
        poses       = torch.stack([c[1] for c in clip], dim=0)  # (self.steps, 3)
        
        if self.train == 'train' and self.use_merge:
            merge_pcs   = torch.cat([c[2] for c in clip], dim=0)  # (self.steps, 1, 251, 251) 
            merge_poses = torch.cat([c[3] for c in clip], dim=0)  # (self.steps, 3)
            
            pcs = torch.cat([pcs, merge_pcs], dim=0)
            poses = torch.cat([poses, merge_poses], dim=0)

        if self.include_vos:
            vos = self.vo_func(poses.unsqueeze(0))[0]
            poses = torch.cat((poses, vos), dim=0)

        batch = {
            "image": pcs,
            "pose": poses,
        }
        return batch

    def __len__(self):
        L = len(self.dset)
        return L