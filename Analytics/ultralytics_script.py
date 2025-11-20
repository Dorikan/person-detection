"""
Скрипт для бенчмаркинга моделей YOLO/RT-DETR на видео с использованием трекинга.
Логирует метрики производительности и результирующее видео в MLflow.

Использование:
    python track_benchmark.py --source video.mp4 --weights yolov8n.pt yolo11n.pt --conf 0.25
"""

import argparse
import os
import sys
import time
import cv2
import mlflow
from pathlib import Path
from typing import List, Optional
from ultralytics import YOLO, RTDETR


def run_benchmark(
        source: str,
        weights_list: List[str],
        tracker: str,
        conf: float,
        iou: float,
        target_fps: Optional[int],
        device: str,
        mlflow_uri: str,
        classes: Optional[List[int]],
        imgsz: Optional[int],
) -> None:
    """
    Основная функция цикла бенчмарка. Прогоняет список моделей по видео.

    Args:
        source (str): Путь к видеофайлу.
        weights_list (List[str]): Список путей к весам моделей (.pt).
        tracker (str): Конфиг трекера ('bytetrack.yaml' или 'botsort.yaml').
        conf (float): Порог уверенности.
        iou (float): Порог IOU.
        target_fps (Optional[int]): Целевой FPS видео (если None, берется оригинал).
        device (str): 'cpu' или 'cuda'.
        mlflow_uri (str): Адрес MLflow сервера.
        classes (Optional[List[int]]): Список классов для детекции.
        imgsz (Optional[int]): Размер кадра для обработки
    """

    mlflow_setup(mlflow_uri)

    if not os.path.exists(source):
        print(f"Ошибка: Файл источника '{source}' не найден.")
        sys.exit(1)

    for weights in weights_list:
        model_name = Path(weights).stem
        run_name = f"{model_name}_{Path(tracker).stem}"
        output_video_path = f"res_{run_name}.mp4"

        print(f"\nЗапуск: {model_name} (Weights: {weights})")

        cap = None
        out = None

        try:
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({
                    "model": model_name,
                    "weights_path": weights,
                    "tracker": tracker,
                    "conf_threshold": conf,
                    "iou_threshold": iou,
                    "device": device,
                    "source": source,
                    'imsize': imgsz
                })

                print(f"Загрузка весов: {weights}...")
                if "rtdetr" in model_name.lower():
                    model = RTDETR(weights)
                else:
                    model = YOLO(weights)

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

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    t_start = time.perf_counter()

                    results = model.track(
                        source=frame,
                        persist=True,
                        tracker=tracker,
                        conf=conf,
                        iou=iou,
                        device=device,
                        classes=classes,
                        imgsz=imgsz,
                        verbose=False
                    )

                    t_end = time.perf_counter()
                    total_inference_time += (t_end - t_start)

                    if results:
                        annotated_frame = results[0].plot(
                            line_width=1,
                            font_size=8,
                            labels=True
                        )
                    else:
                        annotated_frame = frame

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
            print(f"\nОшибка при обработке {model_name}: {e}")
            mlflow.log_param("error", str(e))
            mlflow.end_run(status="FAILED")
        finally:
            if cap is not None and cap.isOpened():
                cap.release()
            if out is not None and out.isOpened():
                out.release()

    print("\nВсе задачи выполнены.")


def mlflow_setup(mlflow_uri: str):
    mlflow.set_tracking_uri(mlflow_uri)
    experiment_name = "Video_Tracking_Benchmark"
    mlflow.set_experiment(experiment_name)

    print(f"MLflow tracking URI: {mlflow_uri}")
    print(f"Experiment: {experiment_name}")


def parse_arguments():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Запуск бенчмарка трекинга YOLO с логированием в MLflow."
    )

    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Путь к видеофайлу для обработки'
    )
    parser.add_argument(
        '--weights',
        nargs='+',
        required=True,
        help='Список путей к весам моделей (например: yolov8n.pt yolo11s.pt)'
    )

    parser.add_argument(
        '--tracker',
        type=str,
        default='bytetrack.yaml',
        help='Тип трекера: bytetrack.yaml или botsort.yaml'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Порог уверенности (Confidence Threshold)'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='Порог IOU для NMS'
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
        default='cpu',
        help='Устройство: cpu, cuda:0, mps'
    )
    parser.add_argument(
        '--mlflow-uri',
        type=str,
        default='http://127.0.0.1:5000',
        help='URI сервера MLflow'
    )
    parser.add_argument(
        '--classes',
        nargs='+',
        type=int,
        default=None,
        help='Фильтр классов по ID (например: 0 2). По умолчанию все.'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Размер до которого модель будет сжимать кадры. По умолчанию 640.'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_benchmark(
        source=args.source,
        weights_list=args.weights,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        target_fps=args.target_fps,
        device=args.device,
        mlflow_uri=args.mlflow_uri,
        classes=args.classes,
        imgsz=args.imgsz,
    )
