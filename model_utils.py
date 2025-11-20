"""
Модуль с архитектурой и логикой инференса модели PairDETR.
Содержит функции модификации голов Deformable DETR и кастомный forward pass.
"""

import torch
import torch.nn as nn
from transformers.models.deformable_detr.modeling_deformable_detr import DeformableDetrMLPPredictionHead
from typing import Dict, Optional


def build_pair_detr_heads(
        model: nn.Module,
        num_queries: int,
        num_classes: int
) -> nn.Module:
    """
    Настраивает головы модели под архитектуру PairDETR.

    Args:
        model: Базовая модель DeformableDetrForObjectDetection.
        num_queries: Количество объектных запросов.
        num_classes: Количество классов для детекции.

    Returns:
        nn.Module: Модифицированная модель.
    """
    in_features = model.class_embed[0].in_features
    model.model.query_position_embeddings = nn.Embedding(num_queries, 512)

    class_embed = nn.Linear(in_features, num_classes)

    bbox_embed = DeformableDetrMLPPredictionHead(
        input_dim=256, hidden_dim=256, output_dim=8, num_layers=3
    )

    model.class_embed = nn.ModuleList([class_embed for _ in range(6)])
    model.bbox_embed = nn.ModuleList([bbox_embed for _ in range(6)])
    return model
