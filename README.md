# BEVDiffLoc
BEVDiffLoc: End-to-End LiDAR Global Localization in BEV View based on Diffusion Model

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

- NCLT: We use [NCLT Sample Python Scripts](https://robots.engin.umich.edu/nclt/) to preprocess velodyne_sync to speed up data reading. We provided within it [nclt_precess.py](preprocess/nclt_precess.py).

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
**TODO**
```bibtex
@article{Wang2025BEVDiffLoc,
	title={BEVDiffLoc: End-to-End LiDAR Global Localization in BEV View based on Diffusion Model},
	author={Z. Wang and C. Shi and N. Wang and Q. Yu and X. Chen and M. Lu},
	year={2025},
 journal={arXiv preprint arXiv:xx.01929}, 
}
```

## Acknowledgement

 We appreciate the code of [DiffLoc](https://github.com/liw95/DiffLoc) and [BEVPlace++](https://github.com/zjuluolun/BEVPlace2) they shared.

