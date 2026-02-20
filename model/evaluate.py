"""
模型评估脚本
计算各项指标: MAE, RMSE, R², F1-Score, AUC-ROC
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from torch.utils.data import DataLoader

from model.dataset import TaihuDataset, collate_fn
from model.graph_builder import load_graph, FEATURE_DIM, WATER_QUALITY_PARAMS
from model.stgat_model import STGAT


def evaluate_model(model, loader, edge_index, edge_weight, device):
    """在测试集上评估模型"""
    model.eval()

    all_wq_preds = []
    all_wq_targets = []
    all_bloom_preds = []
    all_bloom_probs = []
    all_bloom_targets = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y_wq = batch["y_wq"].to(device)
            y_bloom = batch["y_bloom"].to(device)

            wq_pred, wq_log_var, bloom_pred = model(x, edge_index, edge_weight)

            all_wq_preds.append(wq_pred.cpu().numpy())
            all_wq_targets.append(y_wq.cpu().numpy())
            all_bloom_probs.append(torch.softmax(bloom_pred, dim=-1).cpu().numpy())
            all_bloom_preds.append(bloom_pred.argmax(dim=-1).cpu().numpy())
            all_bloom_targets.append(y_bloom.cpu().numpy())

    wq_preds = np.concatenate(all_wq_preds)      # (N_samples, N_nodes, P, 11)
    wq_targets = np.concatenate(all_wq_targets)
    bloom_preds = np.concatenate(all_bloom_preds)  # (N_samples, N_nodes)
    bloom_probs = np.concatenate(all_bloom_probs)  # (N_samples, N_nodes, 4)
    bloom_targets = np.concatenate(all_bloom_targets)

    return wq_preds, wq_targets, bloom_preds, bloom_probs, bloom_targets


def compute_wq_metrics(preds, targets):
    """计算水质预测指标"""
    metrics = {}

    # 全局指标
    for i, param in enumerate(WATER_QUALITY_PARAMS):
        p = preds[:, :, :, i].flatten()
        t = targets[:, :, :, i].flatten()

        # 过滤 NaN
        mask = ~(np.isnan(p) | np.isnan(t))
        p, t = p[mask], t[mask]

        if len(p) == 0:
            continue

        metrics[param] = {
            "mae": float(mean_absolute_error(t, p)),
            "rmse": float(np.sqrt(mean_squared_error(t, p))),
            "r2": float(r2_score(t, p)) if len(np.unique(t)) > 1 else 0.0,
        }

    # 按预测步骤的指标
    num_steps = preds.shape[2]
    step_metrics = {}
    for step in range(num_steps):
        step_preds = preds[:, :, step, :].reshape(-1, preds.shape[-1])
        step_targets = targets[:, :, step, :].reshape(-1, targets.shape[-1])

        step_mae = np.nanmean(np.abs(step_preds - step_targets), axis=0)
        step_metrics[f"day_{step + 1}"] = {
            param: float(step_mae[i]) for i, param in enumerate(WATER_QUALITY_PARAMS)
        }

    return metrics, step_metrics


def compute_bloom_metrics(preds, probs, targets):
    """计算蓝藻预警指标"""
    p_flat = preds.flatten()
    t_flat = targets.flatten()
    prob_flat = probs.reshape(-1, probs.shape[-1])

    metrics = {
        "accuracy": float(accuracy_score(t_flat, p_flat)),
        "f1_macro": float(f1_score(t_flat, p_flat, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(t_flat, p_flat, average="weighted", zero_division=0)),
    }

    # 每个类别的 F1
    class_names = ["无风险", "轻度", "中度", "重度"]
    f1_per_class = f1_score(t_flat, p_flat, average=None, zero_division=0)
    for i, name in enumerate(class_names):
        if i < len(f1_per_class):
            metrics[f"f1_{name}"] = float(f1_per_class[i])

    # AUC-ROC (需要多分类 One-vs-Rest)
    try:
        auc = roc_auc_score(t_flat, prob_flat, multi_class="ovr", average="macro")
        metrics["auc_roc"] = float(auc)
    except Exception:
        metrics["auc_roc"] = None

    # 分类报告
    report = classification_report(
        t_flat, p_flat, target_names=class_names, zero_division=0
    )

    return metrics, report


def main():
    parser = argparse.ArgumentParser(description="TaihuGuard 模型评估")
    parser.add_argument("--checkpoint", required=True, help="模型权重路径")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--graph_dir", default="data/graph")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output", default="weights/evaluation_report.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")

    # 加载 checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    train_args = checkpoint.get("args", {})
    logger.info(f"加载模型: {args.checkpoint}")

    # 加载图结构
    graph = load_graph(args.graph_dir)
    edge_index = torch.LongTensor(graph["edge_index_np"]).to(device)
    edge_weight = torch.FloatTensor(graph["edge_weights_np"]).to(device)

    # 数据集 (测试集)
    test_dataset = TaihuDataset(
        data_dir=args.data_dir, graph_dir=args.graph_dir,
        history_steps=train_args.get("history_steps", 18),
        predict_steps=train_args.get("predict_steps", 7),
        start_date="2025-07-01", end_date="2025-12-31",
        mode="test"
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn
    )

    # V2 模型
    model = STGAT(
        input_dim=FEATURE_DIM,
        hidden_dim=train_args.get("hidden_dim", 256),
        num_heads=train_args.get("num_heads", 8),
        num_wq_params=len(WATER_QUALITY_PARAMS),
        predict_steps=train_args.get("predict_steps", 14),
        num_bloom_classes=4,
        temporal_layers=train_args.get("temporal_layers", 4),
        spatial_layers=train_args.get("spatial_layers", 4),
        ff_dim=train_args.get("ff_dim", 512),
        dropout=0.0
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 评估
    logger.info("开始评估...")
    wq_preds, wq_targets, bloom_preds, bloom_probs, bloom_targets = evaluate_model(
        model, test_loader, edge_index, edge_weight, device
    )

    # 水质指标
    wq_metrics, step_metrics = compute_wq_metrics(wq_preds, wq_targets)
    logger.info("\n=== 水质预测指标 ===")
    for param, m in wq_metrics.items():
        logger.info(f"  {param}: MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}, R²={m['r2']:.4f}")

    # 蓝藻预警指标
    bloom_metrics, bloom_report = compute_bloom_metrics(bloom_preds, bloom_probs, bloom_targets)
    logger.info(f"\n=== 蓝藻预警指标 ===")
    logger.info(f"  Accuracy: {bloom_metrics['accuracy']:.4f}")
    logger.info(f"  F1 (macro): {bloom_metrics['f1_macro']:.4f}")
    logger.info(f"  AUC-ROC: {bloom_metrics.get('auc_roc', 'N/A')}")
    logger.info(f"\n{bloom_report}")

    # 保存报告
    report = {
        "water_quality": wq_metrics,
        "water_quality_by_step": step_metrics,
        "bloom_warning": bloom_metrics,
        "model_info": {
            "checkpoint": args.checkpoint,
            "params": train_args,
            "total_params": sum(p.numel() for p in model.parameters())
        }
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"\n评估报告已保存: {output_path}")


if __name__ == "__main__":
    main()
