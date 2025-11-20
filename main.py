"""
Точка входа для запуска PairDETR на видео.
Выполняет детекцию и сохраняет обработанное видео на диск.
"""

import argparse
import os
import sys
import time
import cv2
import torch
from pathlib import Path
from typing import Optional
from PIL import Image
from transformers import (
    DeformableDetrConfig,
    AutoImageProcessor,
    DeformableDetrForObjectDetection
)

from model_utils import build_pair_detr_heads, custom_forward, box_cxcywh_to_xyxy
from visualization_utils import draw_styled_box


def run_processing(
        source: str,
        weights_path: str,
        conf: float,
        target_fps: Optional[int],
        device: str,
        output_path: Optional[str],
        num_queries: int = 1500,
        num_classes: int = 3
) -> None:
    """
    Основная функция обработки видео.
    """
    if not os.path.exists(source):
        print(f"Ошибка: Видео файл '{source}' не найден.")
        sys.exit(1)
    if not os.path.exists(weights_path):
        print(f"Ошибка: Файл весов '{weights_path}' не найден.")
        sys.exit(1)

    if output_path is None:
        file_stem = Path(source).stem
        output_path = f"{file_stem}_processed.mp4"

    device_obj = torch.device(device)
    print(f"\nЗапуск PairDETR на устройстве: {device}")
    print(f"Вход: {source}")
    print(f"Выход будет сохранен в: {output_path}")

    cap = None
    out = None

    try:
        print("Загрузка модели...")
        try:
            config = DeformableDetrConfig("SenseTime/deformable-detr")

            try:
                processor = AutoImageProcessor.from_pretrained("MTSAIR/PairDETR")
            except Exception:
                print("Процессор MTSAIR не найден, использую facebook/detr-resnet-50")
                processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")

            base_model = DeformableDetrForObjectDetection(config)
            model = build_pair_detr_heads(
                base_model,
                num_queries=num_queries,
                num_classes=num_classes
            )

            checkpoint = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)
            model.to(device_obj)
            model.eval()
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели: {e}")

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise IOError("Не удалось открыть видеофайл.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        save_fps = target_fps if target_fps else orig_fps

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, save_fps, (width, height))

        frame_count = 0
        total_inference_time = 0.0

        print("Начало обработки...")

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
                results = custom_forward(
                    model,
                    inputs["pixel_values"],
                    inputs.get("pixel_mask")
                )

            logits = results["logits"][0]
            boxes_raw = results["pred_boxes"][0]

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
                draw_styled_box(
                    annotated_frame,
                    box1_xyxy.cpu().numpy(),
                    f"S_Cls{label}",
                    score.item(),
                    color=(0, 255, 0)  # Зелёный
                )

                box2_cxcywh = box_8[4:8]
                box2_xyxy = box_cxcywh_to_xyxy(box2_cxcywh) * scale
                draw_styled_box(
                    annotated_frame,
                    box2_xyxy.cpu().numpy(),
                    f"O_Cls{label}",
                    score.item(),
                    (255, 0, 0)  # Синий
                )

            t_end = time.perf_counter()
            total_inference_time += (t_end - t_start)

            out.write(annotated_frame)

            if frame_count % 20 == 0:
                print(f"  -> Кадров обработано: {frame_count}", end='\r')

        # --- 4. Завершение ---
        print(f"\nГотово! Обработано {frame_count} кадров.")

        if frame_count > 0:
            avg_ms = (total_inference_time / frame_count) * 1000
            fps_real = frame_count / total_inference_time
            print(f"⚡ Производительность: {fps_real:.2f} FPS, {avg_ms:.2f} мс/кадр")

        print(f"\nВидео успешно сохранено: {os.path.abspath(output_path)}")

    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")
    except Exception as e:
        print(f"\nОшибка: {e}")
    finally:
        if cap and cap.isOpened(): cap.release()
        if out and out.isOpened(): out.release()


def parse_arguments() -> argparse.Namespace:
    """Парсинг аргументов CLI."""
    parser = argparse.ArgumentParser(description="PairDETR Video Processor")

    parser.add_argument(
        '--source', type=str, required=True,
        help='Путь к входному видео'
    )
    parser.add_argument(
        '--weights', type=str, default='pytorch_model.pth',
        help='Путь к файлу весов'
    )
    parser.add_argument(
        '--conf', type=float, default=0.25,
        help='Порог уверенности'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Имя выходного файла (по умолчанию: source_processed.mp4)'
    )
    parser.add_argument(
        '--target-fps', type=int, default=None,
        help='FPS выходного видео'
    )
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
        help='cpu / cuda'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_processing(
        source=args.source,
        weights_path=args.weights,
        conf=args.conf,
        target_fps=args.target_fps,
        device=args.device,
        output_path=args.output
    )
