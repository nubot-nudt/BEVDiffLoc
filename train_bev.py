"""
@author: Ziyue Wang and Wen Li
@file: train_bev.py
@time: 2025/3/12 14:20
"""

import io
import time
import pstats
import cProfile
import torch.nn as nn

from hydra.utils import instantiate
from collections import OrderedDict
from omegaconf import OmegaConf, DictConfig
from pytorch3d.implicitron.tools import vis_utils
from accelerate import Accelerator, DistributedDataParallelKwargs
from utils.train_util import *
from datasets.composition_bev import MF_bev
from tqdm import tqdm
from tensorboardX import SummaryWriter

writer = SummaryWriter(log_dir='./runs/03_10')

def prefix_with_module(checkpoint):
    prefixed_checkpoint = OrderedDict()
    for key, value in checkpoint.items():
        prefixed_key = "module." + key
        prefixed_checkpoint[prefixed_key] = value
    return prefixed_checkpoint


# Wrapper for cProfile.Profile for easily make optional, turn on/off and printing
class Profiler:
    def __init__(self, active: bool):
        self.c_profiler = cProfile.Profile()
        self.active = active

    def enable(self):
        if self.active:
            self.c_profiler.enable()

    def disable(self):
        if self.active:
            self.c_profiler.disable()

    def print(self):
        if self.active:
            s = io.StringIO()
            sortby = pstats.SortKey.CUMULATIVE
            ps = pstats.Stats(self.c_profiler, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())


def get_thread_count(var_name):
    return os.environ.get(var_name)


def train_fn(cfg: DictConfig):
    # NOTE carefully double check the instruction from huggingface!

    OmegaConf.set_struct(cfg, False)

    # Initialize the accelerator
    accelerator = Accelerator(even_batches=False, device_placement=False)

    accelerator.print("Model Config:")
    accelerator.print(OmegaConf.to_yaml(cfg))

    accelerator.print("Accelerator State:")
    accelerator.print(accelerator.state)

    torch.backends.cudnn.benchmark = cfg.train.cudnnbenchmark

    set_seed_and_print(cfg.seed)

    if accelerator.is_main_process:
        viz = vis_utils.get_visdom_connection(
            server="http://127.0.0.1",
            port=int(os.environ.get("VISDOM_PORT", 8097)),
        )
    
    viz = vis_utils.get_visdom_connection(server="http://127.0.0.1",port=int(os.environ.get("VISDOM_PORT", 8097)))

    accelerator.print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!  OMP_NUM_THREADS: {get_thread_count('OMP_NUM_THREADS')}")
    accelerator.print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!  MKL_NUM_THREADS: {get_thread_count('MKL_NUM_THREADS')}")

    accelerator.print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!  SLURM_CPU_BIND: {get_thread_count('SLURM_CPU_BIND')}")
    accelerator.print(
        f"!!!!!!!!!!!!!!!!!!!!!!!!!!  SLURM_JOB_CPUS_PER_NODE: {get_thread_count('SLURM_JOB_CPUS_PER_NODE')}")

    train_dataset = MF_bev(cfg.train.dataset, cfg, split='train')
    eval_dataset = MF_bev(cfg.train.dataset, cfg, split='eval')

    if cfg.train.num_workers > 0:
        persistent_workers = cfg.train.persistent_workers
    else:
        persistent_workers = False

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train.batch_size,
                                             num_workers=cfg.train.num_workers,
                                             pin_memory=cfg.train.pin_memory,
                                             shuffle=True, drop_last=True,
                                             persistent_workers=persistent_workers
                                             )  # collate_fn
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=cfg.train.batch_size,
                                                  num_workers=cfg.train.num_workers, pin_memory=cfg.train.pin_memory,
                                                  shuffle=False, persistent_workers=persistent_workers)  # collate_fn

    accelerator.print("length of train dataloader is: ", len(dataloader))
    accelerator.print("length of eval dataloader is: ", len(eval_dataloader))

    # Instantiate the model
    model = instantiate(cfg.MODEL, _recursive_=False)
    
    model = model.to(accelerator.device)
    criterion = nn.BCEWithLogitsLoss()

    # Define the numer of epoch
    num_epochs = cfg.train.epochs

    # log
    if os.path.exists(cfg.exp_dir) == 0:
        os.mkdir(cfg.exp_dir)
    # Define the optimizer
    if cfg.train.warmup_sche:
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=cfg.train.lr)
        lr_scheduler = WarmupCosineLR(optimizer=optimizer, lr=cfg.train.lr,
                                      warmup_steps=cfg.train.restart_num * len(dataloader), momentum=0.9,
                                      max_steps=len(dataloader) * (cfg.train.epochs - cfg.train.restart_num))
    else:
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)


    model, dataloader, optimizer, lr_scheduler = accelerator.prepare(model, dataloader, optimizer, lr_scheduler)

    accelerator.print(f"xxxxxxxxxxxxxxxxxx dataloader has {dataloader.num_workers} num_workers")

    start_epoch = 0

    to_plot = ("loss", "lr", "diffloss", "error_t", "error_q")

    stats = VizStats(to_plot)

    for epoch in range(start_epoch, num_epochs):
        stats.new_epoch()

        set_seed_and_print(cfg.seed + epoch)

        # Evaluation
        if (epoch != 0) and (epoch % cfg.train.eval_interval == 0):
        # if (epoch % cfg.train.eval_interval == 0):
            accelerator.print(f"----------Start to eval at epoch {epoch}----------")
            _train_or_eval_fn(model, criterion, eval_dataloader, cfg, optimizer, stats, accelerator, lr_scheduler, epoch, training=False)
            accelerator.print(f"----------Finish the eval at epoch {epoch}----------")
        else:
            accelerator.print(f"----------Skip the eval at epoch {epoch}----------")

        # Training
        accelerator.print(f"----------Start to train at epoch {epoch}----------")
        _train_or_eval_fn(model, criterion, dataloader, cfg, optimizer, stats, accelerator, lr_scheduler, epoch, training=True)
        accelerator.print(f"----------Finish the train at epoch {epoch}----------")

        if accelerator.is_main_process:
            for g in optimizer.param_groups:
                lr = g['lr']
                break
            accelerator.print(f"----------LR is {lr}----------")
            accelerator.print(f"----------Saving stats to {cfg.exp_name}----------")
            stats.update({"lr": lr}, stat_set="train")
            stats.plot_stats(viz=viz, visdom_env=cfg.exp_name)
            accelerator.print(f"----------Done----------")

        if epoch >= 40:
            accelerator.wait_for_everyone()
            ckpt_path = os.path.join(cfg.exp_dir, f"ckpt_{epoch:06}.pth")
            accelerator.print(f"----------Saving the ckpt at epoch {epoch} to {ckpt_path}----------")
            unwrapped_model = accelerator.unwrap_model(model)
            if epoch % 5 == 0:
                accelerator.save(unwrapped_model.state_dict(), ckpt_path)

            if accelerator.is_main_process:
                stats.save(cfg.exp_dir + "stats")

    return True


def _train_or_eval_fn(model, criterion, dataloader, cfg, optimizer, stats, accelerator, lr_scheduler, epoch, training=True):
    if training:
        model.train()
    else:
        model.eval()

    # print(f"Start the loop for process {accelerator.process_index}")

    time_start = time.time()
    max_it = len(dataloader)

    pose_stats = os.path.join(cfg.train.dataroot, cfg.train.dataset, cfg.train.dataset + '_pose_stats.txt')
    pose_m, pose_s = np.loadtxt(pose_stats)
    pose_s = torch.from_numpy(pose_s).to(accelerator.device)
    pose_m = torch.from_numpy(pose_m).to(accelerator.device)

    tqdm_loader = tqdm(dataloader, total=len(dataloader))
    for step, batch in enumerate(tqdm_loader):
        images = batch["image"].to(accelerator.device)  # [B, N, 3, 251, 251]
        batch_size, frame_size = images.size(0), images.size(1)
        poses = batch["pose"].to(accelerator.device)  # [B, N, 3]
        H, W = images.size(-2), images.size(-1)

        if training:
            predictions = model(images, poses, training=True)
            predictions["diffloss"] = predictions["diffloss"]
            loss = predictions["diffloss"]
            writer.add_scalar('train/diffloss', loss.item(), step + epoch * max_it)
        else:
            with torch.no_grad():
                predictions = model(images, training=False)

        # calculate metric
        frame_num = frame_size * batch_size
        pred_poses = predictions['pred_pose'].reshape(frame_num, 3)  # [B*N, 3]
        gt_poses = poses.reshape(frame_num, 3)  # [B*N, 3]

        iou = 0.
        for i in range(frame_num):
            if i == 0:
                error_t = t_error(pred_poses[i, :2], gt_poses[i, :2], pose_s[:2], pose_m[:2])
                error_q = q_error(pred_poses[i, 2], gt_poses[i, 2])
            else:
                error_t += (t_error(pred_poses[i, :2], gt_poses[i, :2], pose_s[:2], pose_m[:2]))
                error_q += (q_error(pred_poses[i, 2], gt_poses[i, 2]))

        predictions['error_t'] = error_t / frame_num
        predictions['error_q'] = error_q / frame_num

        if training:
            writer.add_scalar('train/error_t', predictions['error_t'].item(), step + epoch * max_it)
            writer.add_scalar('train/error_q', predictions['error_q'].item(), step + epoch * max_it)

        if training:
            stats.update(predictions, time_start=time_start, stat_set="train")
            if step % cfg.train.print_interval == 0:
                accelerator.print(stats.print(stat_set="train", max_it=max_it))
        else:
            stats.update(predictions, time_start=time_start, stat_set="eval")
            if step % cfg.train.print_interval == 0:
                accelerator.print(stats.print(stat_set="eval", max_it=max_it))

        if training:
            optimizer.zero_grad()
            accelerator.backward(loss)
            if cfg.train.clip_grad > 0 and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), cfg.train.clip_grad)
            optimizer.step()
            lr_scheduler.step()

    return True

def t_error(pred_poses, gt_poses, pose_s, pose_mean):
    with torch.no_grad():
        error_t = val_translation(pred_poses, gt_poses, pose_s, pose_mean)

    return error_t

def q_error(pred_poses, gt_poses):
    with torch.no_grad():
        p = r_to_d(pred_poses)
        q = r_to_d(gt_poses)
        error_q = abs(p - q)

    return error_q

def r_to_d(r):
    
    d = r * 180 / np.pi
    
    return d

def val_translation(pred_p, gt_p, pose_s, pose_mean):
    """
    test model, compute error (numpy)
    input:
        pred_p: [3,]
        gt_p: [3,]
    returns:
        translation error (m):
    """
    pred_p = (pred_p * pose_s) + pose_mean
    gt_p = (gt_p * pose_s) + pose_mean
    error = torch.linalg.norm(gt_p - pred_p)

    return error


if __name__ == '__main__':
    # oxford_bev.yaml / nclt_bev.yaml
    conf = OmegaConf.load('cfgs/oxford_bev.yaml')
    # conf = OmegaConf.load('cfgs/nclt_bev.yaml')
    train_fn(conf)
