"""
@author: Ziyue Wang and Wen Li
@file: test_bev.py
@time: 2025/3/12 14:20
"""

import time
import matplotlib
import os.path as osp
matplotlib.use('Agg')

from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from utils.train_util import *
from utils.utils import seed_all_random_engines
from utils.pose_util import qexp, val_translation, val_rotation, r_to_d
from datasets.composition_bev import MF_bev
from tensorboardX import SummaryWriter


TOTAL_ITERATIONS = 0

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def test(cfg: DictConfig):
    # NOTE carefully double check the instruction from huggingface!
    global TOTAL_ITERATIONS
    OmegaConf.set_struct(cfg, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instantiate the model
    model = instantiate(cfg.MODEL, _recursive_=False)

    eval_dataset = MF_bev(cfg.train.dataset, cfg, split='eval')

    ckpt_path = os.path.join(cfg.ckpt)
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint, strict=True)
        print(f"Loaded checkpoint from: {ckpt_path}")
    else:
        raise ValueError(f"No checkpoint found at: {ckpt_path}")

    if cfg.train.num_workers > 0:
        persistent_workers = cfg.train.persistent_workers
    else:
        persistent_workers = False

    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=cfg.train.val_batch_size,
                                                  num_workers=cfg.train.num_workers,
                                                  pin_memory=cfg.train.pin_memory,
                                                  persistent_workers=persistent_workers,
                                                  shuffle=False)  # collate

    # Move model and images to the GPU
    model = model.to(device)

    # Evaluation Mode
    model.eval()

    # Seed random engines
    seed_all_random_engines(cfg.seed)

    # pose mean and std
    pose_stats = os.path.join(cfg.train.dataroot, cfg.train.dataset, cfg.train.dataset + '_pose_stats.txt')
    pose_m, pose_s = np.loadtxt(pose_stats)
    pose_m = pose_m[:2]
    pose_s = pose_s[:2]
    # results
    gt_translation = np.zeros((len(eval_dataset), 2))
    pred_translation = np.zeros((len(eval_dataset), 2))
    gt_rotation = np.zeros((len(eval_dataset), 1))
    pred_rotation = np.zeros((len(eval_dataset), 1))
    error_t = np.zeros(len(eval_dataset))
    error_q = np.zeros(len(eval_dataset))

    T1 = time.time()
    
    for step, batch in enumerate(eval_dataloader):
        val_pose = batch["pose"][:, -1, :]
        start_idx = step * cfg.train.val_batch_size
        end_idx = min((step + 1) * cfg.train.val_batch_size, len(eval_dataset))
        gt_translation[start_idx:end_idx, :] = val_pose[:, :2].numpy() * pose_s + pose_m
        gt_rotation[start_idx:end_idx, :] = np.asarray([r_to_d(q).flatten() for q in val_pose[:, 2].numpy()])
        images = batch["image"].to(device)
        with torch.no_grad():
            predictions = model(images, sampling_timesteps=cfg.sampling_timesteps, training=False)
        # predicted pose
        pred = predictions['pred_pose']
        s = pred.size()     # out.shape = [B, N, 6]
        pred_t = pred[..., :2]
        pred_q = pred[..., 2]
        # last frame
        pred_t = pred_t.view(s[0], s[1], 2)
        pred_q = pred_q.view(s[0], s[1], 1)
        pred_t = pred_t[:, -1, :]
        pred_q = pred_q[:, -1, :]

        # RTE / RRE
        pred_translation[start_idx:end_idx, :] = pred_t.cpu().numpy() * pose_s + pose_m
        pred_rotation[start_idx:end_idx, :] = np.asarray([r_to_d(q) for q in pred_q.cpu().numpy()])
        error_t[start_idx:end_idx] = np.asarray([val_translation(p, q) for p, q in zip(pred_translation[start_idx:end_idx, :], gt_translation[start_idx:end_idx, :])])
        error_q[start_idx:end_idx] = np.asarray([abs(p - q).squeeze() for p, q in zip(pred_rotation[start_idx:end_idx, :], gt_rotation[start_idx:end_idx, :])])
        error_q[start_idx:end_idx] = np.where(error_q[start_idx:end_idx] > 180, abs(360 - error_q[start_idx:end_idx]), error_q[start_idx:end_idx])

        log_string('MeanTE(m): %f' % np.mean(error_t[start_idx:end_idx], axis=0))
        log_string('MeanRE(degrees): %f' % np.mean(error_q[start_idx:end_idx], axis=0))
        log_string('MedianTE(m): %f' % np.median(error_t[start_idx:end_idx], axis=0))
        log_string('MedianRE(degrees): %f' % np.median(error_q[start_idx:end_idx], axis=0))

    T2 = time.time()
    print("time:", T2-T1)
    
    mean_ATE = np.mean(error_t)
    mean_ARE = np.mean(error_q)
    median_ATE = np.median(error_t)
    median_ARE = np.median(error_q)

    log_string('Mean Position Error(m): %f' % mean_ATE)
    log_string('Mean Orientation Error(degrees): %f' % mean_ARE)
    log_string('Median Position Error(m): %f' % median_ATE)
    log_string('Median Orientation Error(degrees): %f' % median_ARE)

    val_writer.add_scalar('MeanATE', mean_ATE, TOTAL_ITERATIONS)
    val_writer.add_scalar('MeanARE', mean_ARE, TOTAL_ITERATIONS)
    val_writer.add_scalar('MedianATE', median_ATE, TOTAL_ITERATIONS)
    val_writer.add_scalar('MedianARE', median_ARE, TOTAL_ITERATIONS)

    # save error and trajectory
    real_pose = pred_translation - pose_m
    gt_pose = gt_translation - pose_m
    error_t_filename = osp.join(cfg.exp_dir, 'error_t.txt')
    error_q_filename = osp.join(cfg.exp_dir, 'error_q.txt')
    pred_t_filename = osp.join(cfg.exp_dir, 'pred_t.txt')
    gt_t_filename = osp.join(cfg.exp_dir, 'gt_t.txt')
    pred_q_filename = osp.join(cfg.exp_dir, 'pred_q.txt')
    gt_q_filename = osp.join(cfg.exp_dir, 'gt_q.txt')
    np.savetxt(error_t_filename, error_t, fmt='%8.7f')
    np.savetxt(error_q_filename, error_q, fmt='%8.7f')
    np.savetxt(pred_t_filename, real_pose, fmt='%8.7f')
    np.savetxt(gt_t_filename, gt_pose, fmt='%8.7f')
    np.savetxt(pred_q_filename, pred_rotation, fmt='%8.7f')
    np.savetxt(gt_q_filename, gt_rotation, fmt='%8.7f')


if __name__ == '__main__':
    # oxford_bev.yaml / nclt_bev.yaml / 
    conf = OmegaConf.load('cfgs/oxford_bev.yaml')
    LOG_FOUT = open(os.path.join(conf.exp_dir, 'log.txt'), 'w')
    LOG_FOUT.write(str(conf) + '\n')
    val_writer = SummaryWriter(os.path.join(conf.exp_dir, 'valid'))
    # 5 cpu core
    torch.set_num_threads(5)
    test(conf)
