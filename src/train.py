# 图像被填充到 1280x768（即 24,24 高度填充）
# 以便它们可以分成 60 个 128x128 块。
# 该模型每次前向传递只能看到一个奇异的补丁（即，一个图像有 60 个前向传递和优化步骤）。
# 损失计算（每个补丁）为MSELoss(orig_patch_ij, out_patch_ij)，我们有每张图像的平均损失。

# 所有模型都实现了随机二值化，
# 即编码表示为二进制格式。
# 每个补丁的位数由补丁潜在大小给出
# 压缩后的大小为60 * bits_per_patch / 8 / 1024KB。
import os
import yaml
import argparse
from pathlib import Path

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_loader import ImageFolder720p
from utils import save_imgs

from namespace import Namespace
from logger import Logger

from models.cae_32x32x32_zero_pad_bin import CAE

logger = Logger(__name__, colorize=True)


def train(cfg: Namespace) -> None:
    assert cfg.device == "cpu" or (cfg.device == "cuda" and T.cuda.is_available())

    root_dir = Path(__file__).resolve().parents[1]

    logger.info("training: experiment %s" % (cfg.exp_name))

    # make dir-tree
    exp_dir = root_dir / "experiments" / cfg.exp_name # 拼接目录
    # 创建out、checkpoint、logs三个文件夹
    for d in ["out", "checkpoint", "logs"]:
        os.makedirs(exp_dir / d, exist_ok=True)

    # 读写文件
    cfg.to_file(exp_dir / "train_config.json") 

    # tb tb_writer
    tb_writer = SummaryWriter(exp_dir / "logs")
    logger.info("started tensorboard writer")

    # 实例化对象，加载模型
    model = CAE()
    model.train()
    if cfg.device == "cuda":
        model.cuda()
    logger.info(f"loaded model on {cfg.device}")

    # 加载数据集
    dataloader = DataLoader(
        dataset=ImageFolder720p(cfg.dataset_path),
        batch_size=cfg.batch_size, # train.yaml:16
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
    )
    logger.info(f"loaded dataset from {cfg.dataset_path}")
    
    # 传入学习率等参数进行梯度下降
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)
    # 以mse作为损失
    loss_criterion = nn.MSELoss()

    avg_loss, epoch_avg = 0.0, 0.0
    ts = 0

    # 计算平均损失与平均epoch（平均epoch为什么要计算？）
    # EPOCHS
    for epoch_idx in range(cfg.start_epoch, cfg.num_epochs + 1):
        # BATCHES
        # enumerate枚举
        # 将一个可读取的对象传递给enumerate函数，
        # 他返回一个枚举类型的对象。将其转化为列表后输出元素为元祖（‘序号’，对应元素）的一个列表
        for batch_idx, data in enumerate(dataloader, start=1): # batchindex,data有三项：0是全为0的list，1是图片tensor，2是地址list
            img, patches, _ = data # 三项分别对应data的0,1，2

            if cfg.device == "cuda":
                patches = patches.cuda()

            avg_loss_per_image = 0.0
            # 60个patch前向传播
            # 损失计算（每个补丁）为MSELoss(orig_patch_ij, out_patch_ij)，我们有每张图像的平均损失。
            for i in range(6):
                for j in range(10):
                    optimizer.zero_grad() # 设置梯度为0
                    # 送入分批的单个patch，获取重建后的patch，对比进而获得损失
                    x = patches[:, :, i, j, :, :]
                    y = model(x)
                    loss = loss_criterion(y, x)
                    # 计算图片的平均损失
                    avg_loss_per_image += (1 / 60) * loss.item()

                    loss.backward()
                    optimizer.step()

            avg_loss += avg_loss_per_image
            epoch_avg += avg_loss_per_image

            if batch_idx % cfg.batch_every == 0:
                tb_writer.add_scalar("train/avg_loss", avg_loss / cfg.batch_every, ts)

                for name, param in model.named_parameters():
                    tb_writer.add_histogram(name, param, ts)

                logger.debug(
                    "[%3d/%3d][%5d/%5d] avg_loss: %.8f"
                    % (
                        epoch_idx,
                        cfg.num_epochs,
                        batch_idx,
                        len(dataloader),
                        avg_loss / cfg.batch_every,
                    )
                )

                avg_loss = 0.0
                ts += 1
            # -- end batch every

            if batch_idx % cfg.save_every == 0:
                out = T.zeros(6, 10, 3, 128, 128)
                for i in range(6):
                    for j in range(10):
                        x = patches[0, :, i, j, :, :].unsqueeze(0).cuda()
                        out[i, j] = model(x).cpu().data

                out = np.transpose(out, (0, 3, 1, 4, 2))
                out = np.reshape(out, (768, 1280, 3))
                out = np.transpose(out, (2, 0, 1))

                y = T.cat((img[0], out), dim=2).unsqueeze(0)
                save_imgs(
                    imgs=y,
                    to_size=(3, 768, 2 * 1280),
                    name=exp_dir / f"out/{epoch_idx}_{batch_idx}.png",
                )
            # -- end save every
        # -- end batches

        if epoch_idx % cfg.epoch_every == 0:
            epoch_avg /= len(dataloader) * cfg.epoch_every

            tb_writer.add_scalar(
                "train/epoch_avg_loss",
                avg_loss / cfg.batch_every,
                epoch_idx // cfg.epoch_every,
            )

            logger.info("Epoch avg = %.8f" % epoch_avg)
            epoch_avg = 0.0

            T.save(model.state_dict(), exp_dir / f"checkpoint/model_{epoch_idx}.pth")
        # -- end epoch every
    # -- end epoch

    # save final model
    T.save(model.state_dict(), exp_dir / "model_final.pth")

    # cleaning
    tb_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "rt") as fp:
        cfg = Namespace(**yaml.safe_load(fp))

    train(cfg)
