"""
图结构构建器
将太湖监测站点构建为图结构（节点=站点，边=空间连接）
"""

import json
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.spatial.distance import cdist


# 水质参数列表
WATER_QUALITY_PARAMS = [
    "water_temp", "ph", "do", "conductivity", "turbidity",
    "codmn", "nh3n", "tp", "tn", "chla", "algae_density"
]

# 气象参数列表
WEATHER_PARAMS = [
    "temperature", "precipitation", "wind_speed", "humidity",
    "pressure", "solar_radiation", "cloud"
]

# 遥感参数列表
REMOTE_SENSING_PARAMS = ["ndci", "fai", "lst"]

# 时间编码维度
TIME_ENCODING_DIM = 4  # hour_sin, hour_cos, month_sin, month_cos

# 总特征维度
FEATURE_DIM = len(WATER_QUALITY_PARAMS) + len(WEATHER_PARAMS) + TIME_ENCODING_DIM + len(REMOTE_SENSING_PARAMS)  # 11+7+4+3=25

# 距离阈值 (km)，超过此距离的站点不连边
DISTANCE_THRESHOLD = 50.0


def haversine_distance(lat1, lon1, lat2, lon2):
    """计算两点间的 Haversine 距离 (km)"""
    R = 6371.0  # 地球半径 km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def build_graph(stations_path="data/stations.json", distance_threshold=DISTANCE_THRESHOLD,
                save_dir="data/graph"):
    """
    构建站点图结构

    Returns:
        dict: {
            "num_nodes": int,
            "node_ids": list,
            "node_names": list,
            "node_coords": list of [lat, lon],
            "edge_index": list of [src, dst] pairs,
            "edge_weights": list of float,
            "adjacency": 2D list (邻接矩阵)
        }
    """
    # 加载站点数据
    with open(stations_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stations = data["stations"]
    num_nodes = len(stations)

    node_ids = [s["id"] for s in stations]
    node_names = [s["name"] for s in stations]
    node_coords = np.array([[s["lat"], s["lon"]] for s in stations])
    node_rivers = [s.get("river", "") for s in stations]
    node_basins = [s.get("basin", "") for s in stations]

    logger.info(f"构建图结构: {num_nodes} 个节点")

    # 计算距离矩阵
    dist_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            d = haversine_distance(
                node_coords[i][0], node_coords[i][1],
                node_coords[j][0], node_coords[j][1]
            )
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d

    # 构建边：基于距离阈值 + 同河流/同区域加权
    edge_index = []  # [src, dst]
    edge_weights = []
    adjacency = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = dist_matrix[i][j]
            if dist > distance_threshold:
                continue

            # 基础边权 = 1/距离
            weight = 1.0 / max(dist, 0.1)

            # 同河流加权 ×2
            if node_rivers[i] and node_rivers[i] == node_rivers[j]:
                weight *= 2.0

            # 同水域区域加权 ×1.5
            if node_basins[i] and node_basins[i] == node_basins[j]:
                weight *= 1.5

            # 双向边
            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_weights.append(weight)
            edge_weights.append(weight)
            adjacency[i][j] = weight
            adjacency[j][i] = weight

    logger.info(f"图构建完成: {num_nodes} 节点, {len(edge_index)} 条边 (含双向)")
    logger.info(f"距离阈值: {distance_threshold}km, 平均度: {len(edge_index) / num_nodes:.1f}")

    # 归一化边权
    if edge_weights:
        max_w = max(edge_weights)
        edge_weights = [w / max_w for w in edge_weights]

    graph = {
        "num_nodes": num_nodes,
        "node_ids": node_ids,
        "node_names": node_names,
        "node_coords": node_coords.tolist(),
        "node_rivers": node_rivers,
        "node_basins": node_basins,
        "edge_index": edge_index,
        "edge_weights": edge_weights,
        "adjacency": adjacency.tolist(),
        "distance_matrix": dist_matrix.tolist(),
        "distance_threshold": distance_threshold,
        "feature_dim": FEATURE_DIM
    }

    # 保存
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 保存为 JSON (不含 numpy 数组)
    graph_json = {k: v for k, v in graph.items()}
    with open(save_path / "graph.json", "w", encoding="utf-8") as f:
        json.dump(graph_json, f, ensure_ascii=False, indent=2)

    # 保存 edge_index 和 edge_weight 为 numpy 格式
    if edge_index:
        np.save(save_path / "edge_index.npy", np.array(edge_index, dtype=np.int64).T)
        np.save(save_path / "edge_weights.npy", np.array(edge_weights, dtype=np.float32))
    np.save(save_path / "distance_matrix.npy", dist_matrix)

    logger.info(f"图结构已保存到: {save_path}")
    return graph


def load_graph(graph_dir="data/graph"):
    """加载已构建的图结构"""
    graph_dir = Path(graph_dir)

    with open(graph_dir / "graph.json", "r", encoding="utf-8") as f:
        graph = json.load(f)

    # 加载 numpy 数组
    edge_index_file = graph_dir / "edge_index.npy"
    if edge_index_file.exists():
        graph["edge_index_np"] = np.load(edge_index_file)
        graph["edge_weights_np"] = np.load(graph_dir / "edge_weights.npy")

    graph["distance_matrix_np"] = np.load(graph_dir / "distance_matrix.npy")

    return graph


if __name__ == "__main__":
    graph = build_graph()
    print(f"节点数: {graph['num_nodes']}")
    print(f"边数: {len(graph['edge_index'])}")
    print(f"特征维度: {graph['feature_dim']}")
