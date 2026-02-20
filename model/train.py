"""
V2 ST-GAT 模型训练脚本
在 RunPod GPU (RTX 5090, 32GB VRAM) 上运行

V2 特性:
  - 混合精度训练 (AMP FP16)
  - 梯度累积 (等效大 batch)
  - 更长的训练周期
  - 更丰富的日志和指标
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from model.dataset import TaihuDataset, collate_fn
from model.graph_builder import load_graph, FEATURE_DIM, WATER_QUALITY_PARAMS
from model.stgat_model import STGAT, STGATLoss


def parse_args():
    parser = argparse.ArgumentParser(description="TaihuGuard V2 ST-GAT 训练")
    # 模型参数
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--temporal_layers", type=int, default=4)
    parser.add_argument("--spatial_layers", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    # 数据参数
    parser.add_argument("--history_steps", type=int, default=42, help="7天 × 6步/天")
    parser.add_argument("--predict_steps", type=int, default=14, help="预测14天")
    # 训练参数
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accum_steps", type=int, default=2, help="梯度累积步数")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25, help="Early stopping")
    parser.add_argument("--warmup_epochs", type=int, default=10)
    # 损失权重
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=0.5)
    parser.add_argument("--lambda3", type=float, default=0.1)
    # 路径
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--graph_dir", type=str, default="data/graph")
    parser.add_argument("--save_dir", type=str, default="weights")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    # 精度
    parser.add_argument("--fp16", action="store_true", default=True, help="混合精度训练")
    parser.add_argument("--no_fp16", dest="fp16", action="store_false")
    return parser.parse_args()


def warmup_lr(optimizer, epoch, warmup_epochs, base_lr):
    """线性学习率 warmup"""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def train_epoch(model, loader, criterion, optimizer, edge_index, edge_weight,
                device, scaler, fp16, accum_steps):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_metrics = {}
    num_batches = 0

    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        x = batch["x"].to(device)
        y_wq = batch["y_wq"].to(device)
        y_bloom = batch["y_bloom"].to(device)

        with autocast(enabled=fp16):
            wq_pred, wq_log_var, bloom_pred = model(x, edge_index, edge_weight)
            loss, metrics = criterion(wq_pred, wq_log_var, y_wq, bloom_pred, y_bloom)
            loss = loss / accum_steps

        if fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 梯度累积
        if (i + 1) % accum_steps == 0:
            if fp16:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_metrics = {k: v / max(num_batches, 1) for k, v in total_metrics.items()}
    return avg_loss, avg_metrics


@torch.no_grad()
def validate(model, loader, criterion, edge_index, edge_weight, device, fp16):
    """验证"""
    model.eval()
    total_loss = 0
    total_metrics = {}
    num_batches = 0

    all_wq_preds = []
    all_wq_targets = []
    all_bloom_preds = []
    all_bloom_targets = []

    for batch in loader:
        x = batch["x"].to(device)
        y_wq = batch["y_wq"].to(device)
        y_bloom = batch["y_bloom"].to(device)

        with autocast(enabled=fp16):
            wq_pred, wq_log_var, bloom_pred = model(x, edge_index, edge_weight)
            loss, metrics = criterion(wq_pred, wq_log_var, y_wq, bloom_pred, y_bloom)

        total_loss += loss.item()
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        num_batches += 1

        all_wq_preds.append(wq_pred.float().cpu())
        all_wq_targets.append(y_wq.float().cpu())
        all_bloom_preds.append(bloom_pred.float().cpu())
        all_bloom_targets.append(y_bloom.cpu())

    avg_loss = total_loss / max(num_batches, 1)
    avg_metrics = {k: v / max(num_batches, 1) for k, v in total_metrics.items()}

    # 计算额外指标
    if all_wq_preds:
        wq_p = torch.cat(all_wq_preds)
        wq_t = torch.cat(all_wq_targets)
        bloom_p = torch.cat(all_bloom_preds)
        bloom_t = torch.cat(all_bloom_targets)

        # MAE per parameter
        mae = torch.abs(wq_p - wq_t).mean(dim=[0, 1, 2])
        for i, param in enumerate(WATER_QUALITY_PARAMS):
            if i < mae.shape[0]:
                avg_metrics[f"mae_{param}"] = mae[i].item()

        # 蓝藻 Accuracy
        bloom_acc = (bloom_p.argmax(dim=-1) == bloom_t).float().mean()
        avg_metrics["bloom_accuracy"] = bloom_acc.item()

        # 平均不确定性
        avg_metrics["val_uncertainty"] = avg_metrics.get("mean_uncertainty", 0)

    return avg_loss, avg_metrics


def main():
    args = parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    logger.add(f"logs/train_v2_{time.strftime('%Y%m%d_%H%M%S')}.log", level="INFO")

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        logger.info(f"混合精度: {'启用' if args.fp16 else '关闭'}")

    # 图结构
    graph = load_graph(args.graph_dir)
    edge_index = torch.LongTensor(graph["edge_index_np"]).to(device)
    edge_weight = torch.FloatTensor(graph["edge_weights_np"]).to(device)
    num_nodes = graph["num_nodes"]
    logger.info(f"图: {num_nodes} 节点, {edge_index.shape[1]} 边")

    # 数据集
    logger.info("加载训练集 (2016-2023)...")
    train_dataset = TaihuDataset(
        data_dir=args.data_dir, graph_dir=args.graph_dir,
        history_steps=args.history_steps, predict_steps=args.predict_steps,
        start_date="2016-01-01", end_date="2023-12-31", mode="train"
    )
    logger.info("加载验证集 (2024)...")
    val_dataset = TaihuDataset(
        data_dir=args.data_dir, graph_dir=args.graph_dir,
        history_steps=args.history_steps, predict_steps=args.predict_steps,
        start_date="2024-01-01", end_date="2024-12-31", mode="val"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True
    )

    # V2 模型
    model = STGAT(
        input_dim=FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_wq_params=len(WATER_QUALITY_PARAMS),
        predict_steps=args.predict_steps,
        num_bloom_classes=4,
        temporal_layers=args.temporal_layers,
        spatial_layers=args.spatial_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"V2 模型参数量: {param_count:,} ({param_count * 4 / 1e6:.1f} MB)")
    effective_batch = args.batch_size * args.accum_steps
    logger.info(f"有效 batch_size: {args.batch_size} × {args.accum_steps} = {effective_batch}")

    # 损失函数
    criterion = STGATLoss(
        lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
        predict_steps=args.predict_steps
    ).to(device)

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # 调度器
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)

    # AMP Scaler
    scaler = GradScaler(enabled=args.fp16)

    # 恢复训练
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if args.resume and Path(args.resume).exists():
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info(f"恢复训练 epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")

    # 训练循环
    logger.info(f"\n{'=' * 70}")
    logger.info(f"V2 训练开始: epochs={args.epochs}, hidden={args.hidden_dim}, "
                 f"heads={args.num_heads}, T_layers={args.temporal_layers}, "
                 f"S_layers={args.spatial_layers}")
    logger.info(f"history={args.history_steps} steps, predict={args.predict_steps} days")
    logger.info(f"{'=' * 70}\n")

    history = []

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Warmup
        warmup_lr(optimizer, epoch, args.warmup_epochs, args.lr)

        # 训练
        train_loss, train_m = train_epoch(
            model, train_loader, criterion, optimizer,
            edge_index, edge_weight, device, scaler, args.fp16, args.accum_steps
        )

        # 验证
        val_loss, val_m = validate(
            model, val_loader, criterion, edge_index, edge_weight, device, args.fp16
        )

        # 更新 LR (warmup 之后)
        if epoch >= args.warmup_epochs:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - epoch_start

        # 日志
        logger.info(
            f"Epoch [{epoch+1:3d}/{args.epochs}] "
            f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
            f"LR: {current_lr:.2e} | {elapsed:.0f}s"
        )
        if "bloom_accuracy" in val_m:
            logger.info(
                f"  Bloom Acc: {val_m['bloom_accuracy']:.4f} | "
                f"Uncertainty: {val_m.get('mean_uncertainty', 0):.4f}"
            )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
            "train_metrics": train_m,
            "val_metrics": val_m
        })

        # 保存最佳
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "args": vars(args),
                "graph_info": {"num_nodes": num_nodes, "num_edges": edge_index.shape[1]},
                "model_version": "V2"
            }, save_dir / "stgat_best.pt")
            logger.info(f"  ★ Best model saved (val_loss={best_val_loss:.6f})")
        else:
            patience_counter += 1

        # Checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "args": vars(args)
            }, save_dir / f"checkpoint_epoch{epoch+1}.pt")

        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"\nEarly stopping at epoch {epoch+1}")
            break

    # 保存历史
    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"\n训练完成! Best val_loss: {best_val_loss:.6f}")
    logger.info(f"模型: {save_dir / 'stgat_best.pt'}")


if __name__ == "__main__":
    main()
