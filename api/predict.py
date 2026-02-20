"""
预测服务
封装 ONNX 推理引擎，供 API 调用
"""

from pathlib import Path

from loguru import logger


class PredictionService:
    """预测服务包装器"""

    def __init__(self, model_path="weights/stgat_best.onnx", data_dir="data"):
        self.model_path = Path(model_path)
        self.data_dir = data_dir
        self.engine = None
        self._init_engine()

    def _init_engine(self):
        """初始化推理引擎"""
        if not self.model_path.exists():
            logger.warning(f"模型文件不存在: {self.model_path}，将使用 Demo 数据")
            return

        try:
            from model.inference import TaihuInference
            self.engine = TaihuInference(
                model_path=str(self.model_path),
                data_dir=self.data_dir
            )
            logger.info("推理引擎初始化成功")
        except Exception as e:
            logger.error(f"推理引擎初始化失败: {e}")

    def is_loaded(self):
        """检查模型是否已加载"""
        return self.engine is not None and self.engine.session is not None

    def predict(self):
        """执行推理"""
        if not self.is_loaded():
            logger.warning("推理引擎未就绪")
            return None
        return self.engine.predict()
