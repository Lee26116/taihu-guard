"""
ST-GAT 模型训练脚本
在 RunPod GPU 上运行
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from model.dataset import TaihuDataset, collate_fn
from model.graph_builder import load_graph, FEATURE_DIM, WATER_QUALITY_PARAMS
from model.stgat_model import STGAT, STGATLoss


def parse_args():
    parser = argparse.ArgumentParser(description="TaihuGuard ST-GAT 训练")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--history_steps", type=int, default=18)
    parser.add_argument("--predict_steps", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--graph_dir", type=str, default="data/graph")
    parser.add_argument("--save_dir", type=str, default="weights")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的 checkpoint 路径")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=0.5)
    parser.add_argument("--lambda3", type=float, default=0.1)
    return parser.parse_args()


def train_epoch(model, loader, criterion, optimizer, edge_index, edge_weight, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_metrics = {}
    num_batches = 0

    for batch in loader:
        x = batch["x"].to(device)           # (B, N, T, F)
        y_wq = batch["y_wq"].to(device)     # (B, N, P, 11)
        y_bloom = batch["y_bloom"].to(device)  # (B, N)

        optimizer.zero_grad()

        wq_pred, bloom_pred = model(x, edge_index, edge_weight)

        loss, metrics = criterion(wq_pred, y_wq, bloom_pred, y_bloom)

        loss.backward()
        # 梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_metrics = {k: v / max(num_batches, 1) for k, v in total_metrics.items()}
    return avg_loss, avg_metrics


@torch.no_grad()
def validate(model, loader, criterion, edge_index, edge_weight, device):
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

        wq_pred, bloom_pred = model(x, edge_index, edge_weight)

        loss, metrics = criterion(wq_pred, y_wq, bloom_pred, y_bloom)

        total_loss += loss.item()
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        num_batches += 1

        all_wq_preds.append(wq_pred.cpu())
        all_wq_targets.append(y_wq.cpu())
        all_bloom_preds.append(bloom_pred.cpu())
        all_bloom_targets.append(y_bloom.cpu())

    avg_loss = total_loss / max(num_batches, 1)
    avg_metrics = {k: v / max(num_batches, 1) for k, v in total_metrics.items()}

    # 计算额外指标
    if all_wq_preds:
        wq_preds = torch.cat(all_wq_preds)
        wq_targets = torch.cat(all_wq_targets)
        bloom_preds = torch.cat(all_bloom_preds)
        bloom_targets = torch.cat(all_bloom_targets)

        # MAE per parameter
        mae = torch.abs(wq_preds - wq_targets).mean(dim=[0, 1, 2])
        for i, param in enumerate(WATER_QUALITY_PARAMS):
            avg_metrics[f"mae_{param}"] = mae[i].item()

        # 蓝藻预警 Accuracy
        bloom_correct = (bloom_preds.argmax(dim=-1) == bloom_targets).float().mean()
        avg_metrics["bloom_accuracy"] = bloom_correct.item()

    return avg_loss, avg_metrics


def main():
    args = parse_args()

    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 配置日志
    logger.add(f"logs/train_{time.strftime('%Y%m%d_%H%M%S')}.log", level="INFO")

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"显存: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # 加载图结构
    graph = load_graph(args.graph_dir)
    edge_index = torch.LongTensor(graph["edge_index_np"]).to(device)
    edge_weight = torch.FloatTensor(graph["edge_weights_np"]).to(device)
    num_nodes = graph["num_nodes"]
    logger.info(f"图结构: {num_nodes} 节点, {edge_index.shape[1]} 条边")

    # 数据集
    logger.info("加载训练集...")
    train_dataset = TaihuDataset(
        data_dir=args.data_dir, graph_dir=args.graph_dir,
        history_steps=args.history_steps, predict_steps=args.predict_steps,
        start_date="2021-06-01", end_date="2024-12-31", mode="train"
    )
    logger.info("加载验证集...")
    val_dataset = TaihuDataset(
        data_dir=args.data_dir, graph_dir=args.graph_dir,
        history_steps=args.history_steps, predict_steps=args.predict_steps,
        start_date="2025-01-01", end_date="2025-06-30", mode="val"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )

    # 模型
    model = STGAT(
        input_dim=FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_wq_params=len(WATER_QUALITY_PARAMS),
        predict_steps=args.predict_steps,
        num_bloom_classes=4,
        dropout=args.dropout
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量: {param_count:,} ({param_count * 4 / 1e6:.1f} MB)")

    # 损失函数
    criterion = STGATLoss(
        lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
        predict_steps=args.predict_steps
    ).to(device)

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # 学习率调度
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

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
        logger.info(f"从 epoch {start_epoch} 恢复训练, best_val_loss={best_val_loss:.6f}")

    # 训练循环
    logger.info(f"\n{'=' * 60}")
    logger.info(f"开始训练: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    logger.info(f"{'=' * 60}\n")

    training_history = []

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # 训练
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, edge_index, edge_weight, device
        )

        # 验证
        val_loss, val_metrics = validate(
            model, val_loader, criterion, edge_index, edge_weight, device
        )

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        # 日志
        logger.info(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )
        if "bloom_accuracy" in val_metrics:
            logger.info(f"  蓝藻预警 Acc: {val_metrics['bloom_accuracy']:.4f}")

        # 记录历史
        training_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics
        })

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "args": vars(args),
                "graph_info": {
                    "num_nodes": num_nodes,
                    "num_edges": edge_index.shape[1]
                }
            }, save_dir / "stgat_best.pt")
            logger.info(f"  ★ 保存最佳模型 (val_loss={best_val_loss:.6f})")
        else:
            patience_counter += 1

        # 定期保存 checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "args": vars(args)
            }, save_dir / f"checkpoint_epoch{epoch + 1}.pt")

        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"\nEarly stopping at epoch {epoch + 1} (patience={args.patience})")
            break

    # 保存训练历史
    with open(save_dir / "training_history.json", "w") as f:
        json.dump(training_history, f, indent=2)

    logger.info(f"\n训练完成! 最佳验证损失: {best_val_loss:.6f}")
    logger.info(f"模型保存在: {save_dir / 'stgat_best.pt'}")


if __name__ == "__main__":
    main()
