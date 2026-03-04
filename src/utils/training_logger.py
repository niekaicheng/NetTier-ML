"""
训练记录器 (Training Logger)
=============================
自动记录每次训练的数据、方法、参数、结果和时间。
所有记录保存在 results/training_log.json，每次训练自动追加。

用法:
    from src.utils.training_logger import TrainingLogger

    logger = TrainingLogger(experiment_name="E1_nested_cv")
    logger.set_data_info(dataset="CIC-IDS2017", total_samples=250000, ...)
    logger.set_model_info(model_type="RandomForest", hyperparams={...})
    logger.start()
    # ... 训练 ...
    logger.log_epoch(epoch=1, train_loss=0.5, val_loss=0.4, metrics={...})
    logger.set_results(test_f1=0.80, test_accuracy=0.95, ...)
    logger.finish()
"""

import os
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# 默认日志文件路径
DEFAULT_LOG_PATH = "results/training_log.json"


class TrainingLogger:
    """
    训练记录器 - 自动记录并持久化训练全过程信息。
    
    Features:
    - 记录训练数据来源、样本量、划分比例
    - 记录模型类型、超参数
    - 记录每个 epoch 的指标 (loss, accuracy 等)
    - 记录最终测试结果
    - 记录训练时间 (起止时间、总耗时)
    - 自动追加到 JSON 日志文件
    - 支持生成人类可读的摘要
    """

    def __init__(
        self,
        experiment_name: str,
        log_path: str = DEFAULT_LOG_PATH,
        description: str = "",
    ):
        self.experiment_name = experiment_name
        self.log_path = log_path
        self.description = description
        
        # Run ID: experiment_name + timestamp
        self._run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{experiment_name}_{self._run_timestamp}"
        
        # 训练记录结构
        self.record: Dict[str, Any] = {
            "run_id": self.run_id,
            "experiment_name": experiment_name,
            "description": description,
            "status": "initialized",
            "created_at": datetime.now().isoformat(),
            
            # 数据信息
            "data": {},
            
            # 模型信息
            "model": {},
            
            # 训练配置
            "training_config": {},
            
            # Epoch 日志
            "epoch_log": [],
            
            # 最终结果
            "results": {},
            
            # 时间记录
            "timing": {
                "start_time": None,
                "end_time": None,
                "duration_seconds": None,
                "duration_human": None,
            },
            
            # 输出产物
            "artifacts": [],
        }
        
        self._start_time: Optional[float] = None
        self._epoch_count = 0
        
    # ======================== 数据信息 ========================
    
    def set_data_info(
        self,
        dataset: str = "",
        data_path: str = "",
        total_samples: int = 0,
        train_samples: int = 0,
        val_samples: int = 0,
        test_samples: int = 0,
        num_features: int = 0,
        num_classes: int = 0,
        class_distribution: Optional[Dict[str, int]] = None,
        split_method: str = "",
        split_ratios: str = "",
        **extra
    ):
        """记录训练数据信息"""
        self.record["data"] = {
            "dataset": dataset,
            "data_path": data_path,
            "total_samples": total_samples,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "test_samples": test_samples,
            "num_features": num_features,
            "num_classes": num_classes,
            "class_distribution": class_distribution or {},
            "split_method": split_method,
            "split_ratios": split_ratios,
            **extra
        }
        return self
    
    # ======================== 模型信息 ========================
    
    def set_model_info(
        self,
        model_type: str = "",
        model_name: str = "",
        architecture: str = "",
        hyperparams: Optional[Dict[str, Any]] = None,
        total_params: int = 0,
        trainable_params: int = 0,
        framework: str = "",
        **extra
    ):
        """记录模型信息"""
        self.record["model"] = {
            "model_type": model_type,
            "model_name": model_name,
            "architecture": architecture,
            "hyperparams": _serialize(hyperparams or {}),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "framework": framework,
            **extra
        }
        return self
    
    # ======================== 训练配置 ========================
    
    def set_training_config(
        self,
        epochs: int = 0,
        batch_size: int = 0,
        learning_rate: float = 0.0,
        optimizer: str = "",
        scheduler: str = "",
        loss_function: str = "",
        device: str = "",
        early_stopping: bool = False,
        early_stopping_patience: int = 0,
        cv_folds: int = 0,
        **extra
    ):
        """记录训练配置"""
        self.record["training_config"] = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "loss_function": loss_function,
            "device": device,
            "early_stopping": early_stopping,
            "early_stopping_patience": early_stopping_patience,
            "cv_folds": cv_folds,
            **extra
        }
        return self
    
    # ======================== 计时 ========================
    
    def start(self):
        """标记训练开始"""
        self._start_time = time.time()
        self.record["status"] = "running"
        self.record["timing"]["start_time"] = datetime.now().isoformat()
        print(f"[TrainingLogger] Run '{self.run_id}' started at {self.record['timing']['start_time']}")
        return self
    
    def finish(self, status: str = "completed"):
        """标记训练结束并保存记录"""
        end_time = time.time()
        self.record["status"] = status
        self.record["timing"]["end_time"] = datetime.now().isoformat()
        
        if self._start_time is not None:
            duration = end_time - self._start_time
            self.record["timing"]["duration_seconds"] = round(duration, 2)
            self.record["timing"]["duration_human"] = _format_duration(duration)
        
        self.record["timing"]["total_epochs_completed"] = self._epoch_count
        
        # 保存到文件
        self._save()
        
        print(f"[TrainingLogger] Run '{self.run_id}' {status}. "
              f"Duration: {self.record['timing'].get('duration_human', 'N/A')}. "
              f"Log saved to {self.log_path}")
        return self
    
    # ======================== Epoch 日志 ========================
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        train_acc: Optional[float] = None,
        val_acc: Optional[float] = None,
        learning_rate: Optional[float] = None,
        **extra_metrics
    ):
        """记录单个 Epoch 的指标"""
        self._epoch_count = epoch
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "learning_rate": learning_rate,
            "timestamp": datetime.now().isoformat(),
            **extra_metrics
        }
        # Remove None values for cleaner output
        entry = {k: v for k, v in entry.items() if v is not None}
        self.record["epoch_log"].append(entry)
        return self
    
    # ======================== CV Fold 日志 ========================
    
    def log_cv_fold(self, fold: int, metrics: Dict[str, Any], best_params: Optional[Dict] = None):
        """记录交叉验证的某个 Fold 结果"""
        entry = {
            "fold": fold,
            "metrics": _serialize(metrics),
            "best_params": _serialize(best_params) if best_params else {},
            "timestamp": datetime.now().isoformat(),
        }
        self.record["epoch_log"].append(entry)
        self._epoch_count = fold
        return self
    
    # ======================== 最终结果 ========================
    
    def set_results(self, **metrics):
        """
        记录最终测试/评估结果。
        
        示例:
            logger.set_results(
                test_f1_macro=0.80,
                test_precision=0.82,
                test_recall=0.79,
                test_accuracy=0.95,
                best_val_accuracy=0.94,
                classification_report="...",
            )
        """
        self.record["results"] = _serialize(metrics)
        return self
    
    # ======================== 产物记录 ========================
    
    def add_artifact(self, path: str, artifact_type: str = "model", description: str = ""):
        """记录训练输出的文件产物 (模型、报告、图表等)"""
        self.record["artifacts"].append({
            "path": path,
            "type": artifact_type,
            "description": description,
        })
        return self
    
    # ======================== 持久化 ========================
    
    def _save(self):
        """保存当前记录到 JSON 日志文件 (追加模式)"""
        os.makedirs(os.path.dirname(self.log_path) if os.path.dirname(self.log_path) else ".", exist_ok=True)
        
        # 读取已有记录
        existing: List[Dict] = []
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if not isinstance(existing, list):
                    existing = [existing]  # 兼容旧格式
            except (json.JSONDecodeError, Exception):
                existing = []
        
        # 检查是否已有同 run_id 的记录 (更新而非重复追加)
        updated = False
        for i, rec in enumerate(existing):
            if rec.get("run_id") == self.run_id:
                existing[i] = self.record
                updated = True
                break
        
        if not updated:
            existing.append(self.record)
        
        # 写入
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False, default=str)
    
    # ======================== 查询与摘要 ========================
    
    @staticmethod
    def load_all(log_path: str = DEFAULT_LOG_PATH) -> List[Dict]:
        """加载所有历史训练记录"""
        if not os.path.exists(log_path):
            return []
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    
    @staticmethod
    def get_summary(log_path: str = DEFAULT_LOG_PATH) -> str:
        """生成人类可读的训练历史摘要"""
        records = TrainingLogger.load_all(log_path)
        if not records:
            return "No training records found."
        
        lines = []
        lines.append("=" * 90)
        lines.append(f"  TRAINING LOG SUMMARY  ({len(records)} runs)")
        lines.append("=" * 90)
        
        for i, rec in enumerate(records):
            lines.append("")
            lines.append(f"--- Run {i+1}: {rec.get('run_id', 'N/A')} ---")
            lines.append(f"  Experiment : {rec.get('experiment_name', 'N/A')}")
            lines.append(f"  Status     : {rec.get('status', 'N/A')}")
            lines.append(f"  Description: {rec.get('description', '')}")
            
            # 数据
            data = rec.get("data", {})
            if data:
                lines.append(f"  Dataset    : {data.get('dataset', 'N/A')}")
                lines.append(f"  Samples    : Train={data.get('train_samples', '?')} | "
                            f"Val={data.get('val_samples', '?')} | "
                            f"Test={data.get('test_samples', '?')} | "
                            f"Total={data.get('total_samples', '?')}")
                lines.append(f"  Features   : {data.get('num_features', '?')} | Classes: {data.get('num_classes', '?')}")
            
            # 模型
            model = rec.get("model", {})
            if model:
                lines.append(f"  Model      : {model.get('model_type', 'N/A')} ({model.get('model_name', '')})")
                if model.get("total_params"):
                    lines.append(f"  Params     : {model['total_params']:,}")
                hp = model.get("hyperparams", {})
                if hp:
                    hp_str = ", ".join(f"{k}={v}" for k, v in hp.items())
                    lines.append(f"  Hyperparams: {hp_str}")
            
            # 训练配置
            cfg = rec.get("training_config", {})
            if cfg:
                cfg_parts = []
                if cfg.get("epochs"): cfg_parts.append(f"Epochs={cfg['epochs']}")
                if cfg.get("batch_size"): cfg_parts.append(f"BS={cfg['batch_size']}")
                if cfg.get("learning_rate"): cfg_parts.append(f"LR={cfg['learning_rate']}")
                if cfg.get("optimizer"): cfg_parts.append(f"Opt={cfg['optimizer']}")
                if cfg.get("device"): cfg_parts.append(f"Device={cfg['device']}")
                if cfg.get("cv_folds"): cfg_parts.append(f"CV={cfg['cv_folds']}-fold")
                if cfg_parts:
                    lines.append(f"  Config     : {' | '.join(cfg_parts)}")
            
            # 结果
            results = rec.get("results", {})
            if results:
                res_parts = []
                for k, v in results.items():
                    if isinstance(v, float):
                        res_parts.append(f"{k}={v:.4f}")
                    elif isinstance(v, str) and len(v) > 100:
                        res_parts.append(f"{k}=[详见日志]")
                    else:
                        res_parts.append(f"{k}={v}")
                lines.append(f"  Results    : {' | '.join(res_parts)}")
            
            # 时间
            timing = rec.get("timing", {})
            if timing:
                lines.append(f"  Time       : {timing.get('duration_human', 'N/A')} "
                            f"({timing.get('start_time', '?')} → {timing.get('end_time', '?')})")
                if timing.get("total_epochs_completed"):
                    lines.append(f"  Epochs Done: {timing['total_epochs_completed']}")
            
            # 产物
            artifacts = rec.get("artifacts", [])
            if artifacts:
                lines.append(f"  Artifacts  :")
                for a in artifacts:
                    lines.append(f"    - [{a.get('type', '?')}] {a.get('path', '?')} {a.get('description', '')}")
        
        lines.append("")
        lines.append("=" * 90)
        return "\n".join(lines)
    
    @staticmethod
    def print_summary(log_path: str = DEFAULT_LOG_PATH):
        """打印训练历史摘要"""
        print(TrainingLogger.get_summary(log_path))
    
    @staticmethod
    def get_latest(experiment_name: str = None, log_path: str = DEFAULT_LOG_PATH) -> Optional[Dict]:
        """获取最新一条记录 (可按实验名过滤)"""
        records = TrainingLogger.load_all(log_path)
        if experiment_name:
            records = [r for r in records if r.get("experiment_name") == experiment_name]
        return records[-1] if records else None


# ======================== 辅助函数 ========================

def _format_duration(seconds: float) -> str:
    """将秒数格式化为人类可读字符串"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.1f}s"
    else:
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{int(h)}h {int(m)}m {s:.1f}s"


def _serialize(obj: Any) -> Any:
    """将对象转为 JSON 可序列化格式"""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(item) for item in obj]
    if isinstance(obj, (int, float, str, bool)):
        return obj
    # numpy types
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return str(obj)
