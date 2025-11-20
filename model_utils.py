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

def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Вычисляет обратную сигмоиду. Используется для преобразования референсных точек.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    Конвертирует боксы из формата (center_x, center_y, w, h) в (x1, y1, x2, y2).
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def custom_forward(
        model: nn.Module,
        pixel_values: torch.Tensor,
        pixel_mask: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """
    Кастомный проход (forward pass) для PairDETR.
    Корректно обрабатывает слои декодера для получения финальных предсказаний.
    """
    outputs = model.model(
        pixel_values,
        pixel_mask=pixel_mask,
        return_dict=True,
    )

    hidden_states = outputs.last_hidden_state
    init_reference = outputs.init_reference_points
    inter_references = outputs.intermediate_reference_points

    level = len(model.bbox_embed) - 1

    reference = inter_references[:, level - 1]
    reference = inverse_sigmoid(reference)

    outputs_class = model.class_embed[level](hidden_states)
    delta_bbox = model.bbox_embed[level](hidden_states)

    if reference.shape[-1] == 4:
        delta_bbox[..., :4] += reference
    elif reference.shape[-1] == 2:
        cons = inverse_sigmoid(init_reference)
        delta_bbox[..., :2] += reference
        delta_bbox[..., 4:6] += cons

    outputs_coord = delta_bbox.sigmoid()

    return {
        "logits": outputs_class.softmax(dim=-1),
        "pred_boxes": outputs_coord
    }
