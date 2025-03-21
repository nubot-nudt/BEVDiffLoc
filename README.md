# BEVDiffLoc
BEVDiffLoc: End-to-End LiDAR Global Localization in BEV View based on Diffusion Model

# Visualization
![image](img/oxford.gif) ![image](img/nclt.gif)

# Environment

- python 3.10

- pytorch 2.1.2

- cuda 12.1

```
source install.sh
```

## Dataset

We support the [Oxford Radar RobotCar](https://oxford-robotics-institute.github.io/radar-robotcar-dataset/datasets) and [NCLT](https://robots.engin.umich.edu/nclt/) datasets right now.

The data of the Oxford and NCLT dataset should be organized as follows:

```
data_root
├── 2019-01-11-14-02-26-radar-oxford-10k
│   ├── velodyne_left
│   │   ├── xxx.bin
│   │   ├── xxx.bin
│   ├── gps
│   │   ├── gps.csv
│   │   ├── ins.csv
│   ├── velodyne_left.timestamps
│   ├── merge_bev (Data prepare)
│   │   ├── xxx.png
│   │   ├── xxx.png
│   ├── merge_bev.txt (Data prepare)
├── Oxford_pose_stats.txt
├── train_split.txt
├── valid_split.txt
```

## Data prepare

- Oxford&NCLT: We use [merge_nclt.py](merge_nclt.py) and [merge_oxford.py](merge_oxford.py) to generate local scenes for data augmentation.


## Run

### Download the pretrained ViT model
We initialize BEVDiffLoc's feature learner with [DINOv2](https://github.com/facebookresearch/dinov2?tab=readme-ov-file).

### Train

```
accelerate launch --num_processes 1 --mixed_precision fp16 train_bev.py
```

### Test
```
python test_bev.py
```

## Citation

If you find this work helpful, please consider citing:
```bibtex
@article{wang2025bevdiffloc,
  title={BEVDiffLoc: End-to-End LiDAR Global Localization in BEV View based on Diffusion Model},
  author={Wang, Ziyue and Shi, Chenghao and Wang, Neng and Yu, Qinghua and Chen, Xieyuanli and Lu, Huimin},
  journal={arXiv preprint arXiv:2503.11372},
  year={2025}
}
```

## Acknowledgement

 We appreciate the code of [DiffLoc](https://github.com/liw95/DiffLoc) and [BEVPlace++](https://github.com/zjuluolun/BEVPlace2) they shared.

