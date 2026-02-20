"""
一次性脚本: 构建太湖监测站点图结构
输出: data/graph/ 下的 graph.json, edge_index.npy, edge_weights.npy
"""

import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.graph_builder import build_graph


def main():
    import argparse
    parser = argparse.ArgumentParser(description="构建太湖站点图结构")
    parser.add_argument("--stations", default="data/stations.json", help="站点数据路径")
    parser.add_argument("--threshold", type=float, default=50.0, help="距离阈值 (km)")
    parser.add_argument("--output", default="data/graph", help="输出目录")
    args = parser.parse_args()

    graph = build_graph(
        stations_path=args.stations,
        distance_threshold=args.threshold,
        save_dir=args.output
    )

    print(f"\n图结构构建完成:")
    print(f"  节点数: {graph['num_nodes']}")
    print(f"  边数:   {len(graph['edge_index'])}")
    print(f"  特征维: {graph['feature_dim']}")
    print(f"  距离阈: {args.threshold} km")
    print(f"  保存至: {args.output}/")


if __name__ == "__main__":
    main()
