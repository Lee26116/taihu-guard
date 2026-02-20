"""
模型导出: PyTorch → ONNX
导出的 ONNX 模型用于 CPU 推理
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from model.graph_builder import load_graph, FEATURE_DIM, WATER_QUALITY_PARAMS
from model.stgat_model import STGAT


def export_to_onnx(checkpoint_path, output_path, graph_dir="data/graph",
                   history_steps=18, opset_version=14):
    """将 PyTorch 模型导出为 ONNX"""

    device = torch.device("cpu")

    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_args = checkpoint.get("args", {})
    logger.info(f"加载模型: {checkpoint_path}")

    # 加载图结构
    graph = load_graph(graph_dir)
    num_nodes = graph["num_nodes"]
    edge_index = torch.LongTensor(graph["edge_index_np"])
    edge_weight = torch.FloatTensor(graph["edge_weights_np"])

    # 创建模型
    hidden_dim = train_args.get("hidden_dim", 64)
    num_heads = train_args.get("num_heads", 4)
    predict_steps = train_args.get("predict_steps", 7)

    model = STGAT(
        input_dim=FEATURE_DIM,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_wq_params=len(WATER_QUALITY_PARAMS),
        predict_steps=predict_steps,
        num_bloom_classes=4,
        dropout=0.0
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 创建示例输入
    batch_size = 1
    sample_x = torch.randn(batch_size, num_nodes, history_steps, FEATURE_DIM)

    # 测试前向传播
    with torch.no_grad():
        wq_out, bloom_out = model(sample_x, edge_index, edge_weight)
        logger.info(f"前向测试: wq={wq_out.shape}, bloom={bloom_out.shape}")

    # 包装模型（将 edge_index/edge_weight 固化到模型内部）
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, model, edge_index, edge_weight):
            super().__init__()
            self.model = model
            self.register_buffer("edge_index", edge_index)
            self.register_buffer("edge_weight", edge_weight)

        def forward(self, x):
            return self.model(x, self.edge_index, self.edge_weight)

    wrapped_model = ONNXWrapper(model, edge_index, edge_weight)
    wrapped_model.eval()

    # 导出 ONNX
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"导出 ONNX: opset={opset_version}")
    torch.onnx.export(
        wrapped_model,
        sample_x,
        str(output_path),
        opset_version=opset_version,
        input_names=["node_features"],
        output_names=["water_quality_pred", "algal_bloom_risk"],
        dynamic_axes={
            "node_features": {0: "batch_size"},
            "water_quality_pred": {0: "batch_size"},
            "algal_bloom_risk": {0: "batch_size"}
        }
    )

    # 验证 ONNX 模型
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX 模型验证通过")
    except ImportError:
        logger.warning("未安装 onnx 包，跳过验证")
    except Exception as e:
        logger.error(f"ONNX 验证失败: {e}")

    # 测试 ONNX 推理
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(output_path))
        ort_inputs = {"node_features": sample_x.numpy()}
        ort_outputs = session.run(None, ort_inputs)
        logger.info(f"ONNX 推理测试: wq={ort_outputs[0].shape}, bloom={ort_outputs[1].shape}")

        # 比较精度
        wq_diff = np.abs(ort_outputs[0] - wq_out.numpy()).max()
        bloom_diff = np.abs(ort_outputs[1] - bloom_out.numpy()).max()
        logger.info(f"精度对比: wq max_diff={wq_diff:.6f}, bloom max_diff={bloom_diff:.6f}")
    except ImportError:
        logger.warning("未安装 onnxruntime，跳过推理测试")

    # 文件大小
    size_mb = output_path.stat().st_size / 1e6
    logger.info(f"ONNX 模型大小: {size_mb:.1f} MB")
    if size_mb > 20:
        logger.warning(f"模型超过 20MB 目标! ({size_mb:.1f} MB)")

    logger.info(f"导出完成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="导出 ONNX 模型")
    parser.add_argument("--checkpoint", required=True, help="PyTorch checkpoint 路径")
    parser.add_argument("--output", default="weights/stgat_best.onnx")
    parser.add_argument("--graph_dir", default="data/graph")
    parser.add_argument("--opset", type=int, default=14)
    args = parser.parse_args()

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        graph_dir=args.graph_dir,
        opset_version=args.opset
    )


if __name__ == "__main__":
    main()
