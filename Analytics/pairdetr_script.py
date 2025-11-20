"""
Скрипт для бенчмаркинга кастомной модели PairDETR (на базе Deformable DETR).
Логирует метрики производительности и результирующее видео в MLflow.
"""

import argparse
import os
import sys
import time
import cv2
import mlflow
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict
from PIL import Image
from transformers import DeformableDetrConfig, AutoImageProcessor, DeformableDetrForObjectDetection
from transformers.models.deformable_detr.modeling_deformable_detr import DeformableDetrMLPPredictionHead


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Обратная сигмоида для преобразования координат.

    Args:
        x (torch.Tensor): Входной тензор.
        eps (float): Эпсилон для стабильности логарифма.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def build_pair_detr_heads(model: nn.Module, num_queries: int, num_classes: int) -> nn.Module:
    """
    Модифицирует головы модели Deformable DETR под архитектуру PairDETR.

    Args:
        model (nn.Module): Базовая модель DeformableDetrForObjectDetection.
        num_queries (int): Количество запросов (queries).
        num_classes (int): Количество классов.

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


def custom_forward(model: nn.Module, pixel_values: torch.Tensor, pixel_mask: Optional[torch.Tensor] = None) -> Dict[
    str, torch.Tensor]:
    """
    Кастомный forward pass для PairDETR.

    Args:
        model (nn.Module): Модель PairDETR.
        pixel_values (torch.Tensor): Тензор изображения.
        pixel_mask (Optional[torch.Tensor]): Маска пикселей (для паддинга).

    Returns:
        Dict[str, torch.Tensor]: Словарь с ключами 'logits' и 'pred_boxes'.
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


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    Конвертация координат из формата (cx, cy, w, h) в (x1, y1, x2, y2).
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def draw_styled_detections(img: np.ndarray, box: np.ndarray, label_text: str, score: float,
                           color: tuple = (0, 255, 0)) -> None:
    """
    Рисует один бокс с тонкими линиями и полупрозрачной подложкой под текст.
    Модифицирует изображение inplace.
    """
    h_img, w_img, _ = img.shape
    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

    text = f"{label_text} {score:.2f}"
    font_scale = 0.4
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    rx1, ry1 = x1, y1 - text_h - 4
    rx2, ry2 = x1 + text_w + 4, y1

    if ry1 < 0:
        ry1, ry2 = y1, y1 + text_h + 4

    rx1 = max(0, rx1)
    ry1 = max(0, ry1)
    rx2 = min(w_img, rx2)
    ry2 = min(h_img, ry2)

    if rx2 > rx1 and ry2 > ry1:
        roi = img[ry1:ry2, rx1:rx2]
        white = np.full(roi.shape, 255, dtype=np.uint8)
        cv2.addWeighted(roi, 0.6, white, 0.4, 1.0, dst=roi)
        img[ry1:ry2, rx1:rx2] = roi

    ty = ry2 - 3 if ry1 < y1 else ry2 - 3
    cv2.putText(img, text, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)


def run_benchmark(
        source: str,
        weights_path: str,
        conf: float,
        target_fps: Optional[int],
        device: str,
        mlflow_uri: str,
        num_queries: int = 1500,
        num_classes: int = 3
) -> None:
    """
    Основная функция цикла бенчмарка для PairDETR.

    Args:
        source (str): Путь к видеофайлу.
        weights_path (str): Путь к файлу весов (.pth).
        conf (float): Порог уверенности.
        target_fps (Optional[int]): Целевой FPS видео.
        device (str): 'cpu' или 'cuda'.
        mlflow_uri (str): Адрес MLflow сервера.
        num_queries (int): Количество запросов модели.
        num_classes (int): Количество классов модели.
    """

    mlflow_setup(mlflow_uri)

    if not os.path.exists(source):
        print(f"Ошибка: Файл источника '{source}' не найден.")
        sys.exit(1)

    if not os.path.exists(weights_path):
        print(f"Ошибка: Файл весов '{weights_path}' не найден.")
        sys.exit(1)

    model_name = "PairDETR_Custom"
    run_name = f"{model_name}_{Path(weights_path).stem}"
    output_video_path = f"res_{run_name}.mp4"

    print(f"\nЗапуск: {model_name} (Weights: {weights_path})")

    cap = None
    out = None
    device_obj = torch.device(device)

    try:
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "model": model_name,
                "weights_path": weights_path,
                "conf_threshold": conf,
                "device": device,
                "source": source,
                "num_queries": num_queries,
                "num_classes": num_classes
            })

            print("Конфигурация и загрузка модели...")
            try:
                config = DeformableDetrConfig("SenseTime/deformable-detr")
                try:
                    processor = AutoImageProcessor.from_pretrained("MTSAIR/PairDETR")
                except Exception:
                    print("⚠️ Не удалось загрузить процессор MTSAIR/PairDETR, использую базовый.")
                    processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")

                base_model = DeformableDetrForObjectDetection(config)
                model = build_pair_detr_heads(base_model, num_queries=num_queries, num_classes=num_classes)

                print(f"Загрузка state_dict...")
                checkpoint = torch.load(weights_path, map_location="cpu")
                model.load_state_dict(checkpoint, strict=False)
                model.to(device_obj)
                model.eval()

            except Exception as e:
                raise RuntimeError(f"Ошибка инициализации модели: {e}")

            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise IOError(f"Не удалось открыть видеопоток: {source}")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            orig_fps = cap.get(cv2.CAP_PROP_FPS)
            save_fps = target_fps if target_fps else orig_fps

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, save_fps, (width, height))

            frame_count = 0
            total_inference_time = 0.0

            print("Начало обработки кадров...")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                t_start = time.perf_counter()

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)

                inputs = processor(images=pil_img, return_tensors="pt")
                inputs = {k: v.to(device_obj) for k, v in inputs.items()}

                with torch.no_grad():
                    results = custom_forward(model, inputs["pixel_values"], inputs.get("pixel_mask"))

                logits = results["logits"][0]  # [Queries, Classes]
                boxes_raw = results["pred_boxes"][0]  # [Queries, 8]

                probs, labels = logits.max(dim=-1)
                keep = probs > conf

                keep_probs = probs[keep]
                keep_labels = labels[keep]
                keep_boxes = boxes_raw[keep]

                annotated_frame = frame.copy()
                scale = torch.tensor([width, height, width, height], device=device_obj)

                for score, label, box_8 in zip(keep_probs, keep_labels, keep_boxes):
                    if label == 2:
                        continue

                    box1_cxcywh = box_8[:4]
                    box1_xyxy = box_cxcywh_to_xyxy(box1_cxcywh) * scale
                    draw_styled_detections(
                        annotated_frame,
                        box1_xyxy.cpu().numpy(),
                        f"S_Cls{label}",
                        score.item(),
                        color=(0, 255, 0) # Зелёный
                    )

                    box2_cxcywh = box_8[4:8]
                    box2_xyxy = box_cxcywh_to_xyxy(box2_cxcywh) * scale
                    draw_styled_detections(
                        annotated_frame,
                        box2_xyxy.cpu().numpy(),
               f"O_Cls{label}",
                        score.item(),
                        (255, 0, 0)  # Синий
                    )

                t_end = time.perf_counter()
                total_inference_time += (t_end - t_start)

                out.write(annotated_frame)

                if frame_count % 50 == 0:
                    print(f"  -> Обработано кадров: {frame_count}", end='\r')

            if frame_count > 0:
                avg_inference_ms = (total_inference_time / frame_count) * 1000
                processed_fps = frame_count / total_inference_time

                print(f"\nГотово. FPS обработки: {processed_fps:.2f}, Time/Frame: {avg_inference_ms:.2f}ms")

                mlflow.log_metric("processing_fps", processed_fps)
                mlflow.log_metric("avg_inference_time_ms", avg_inference_ms)
                mlflow.log_metric("total_frames", frame_count)

            cap.release()
            out.release()

            if os.path.exists(output_video_path):
                print(f"Загрузка артефакта в MLflow...")
                mlflow.log_artifact(output_video_path)
                os.remove(output_video_path)

    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        return
    except Exception as e:
        print(f"\nОшибка при обработке: {e}")
        mlflow.log_param("error", str(e))
        mlflow.end_run(status="FAILED")
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
        if out is not None and out.isOpened():
            out.release()

    print("\nВсе задачи выполнены.")


def mlflow_setup(mlflow_uri: str):
    """
    Функция настройки соединения с MLflow.

    Args:
        mlflow_uri (str): Ссылка на MLflow сервер.
    """
    mlflow.set_tracking_uri(mlflow_uri)
    experiment_name = "PairDETR_Benchmark"
    mlflow.set_experiment(experiment_name)

    print(f"MLflow tracking URI: {mlflow_uri}")
    print(f"Experiment: {experiment_name}")


def parse_arguments():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Запуск бенчмарка PairDETR с логированием в MLflow."
    )

    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Путь к видеофайлу для обработки'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='pytorch_model.pth',
        help='Путь к файлу весов (.pth)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Порог уверенности (Confidence Threshold)'
    )
    parser.add_argument(
        '--target-fps',
        type=int,
        default=None,
        help='FPS результирующего видео (по умолчанию = исходному)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Устройство: cpu, cuda:0'
    )
    parser.add_argument(
        '--mlflow-uri',
        type=str,
        default='http://127.0.0.1:5000',
        help='URI сервера MLflow'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_benchmark(
        source=args.source,
        weights_path=args.weights,
        conf=args.conf,
        target_fps=args.target_fps,
        device=args.device,
        mlflow_uri=args.mlflow_uri
    )
