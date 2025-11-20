"""
Модуль визуализации результатов детекции.
Содержит функции для отрисовки боксов и текста с полупрозрачной подложкой.
"""

import cv2
import numpy as np
from typing import Tuple


def draw_styled_box(
        img: np.ndarray,
        box: np.ndarray,
        label_text: str,
        score: float,
        color: Tuple[int, int, int] = (0, 255, 0)
) -> None:
    """
    Рисует один ограничивающий бокс (bbox) с тонкими линиями и
    полупрозрачной подложкой под текст.

    Args:
        img (np.ndarray): Исходное изображение (BGR), изменяется inplace.
        box (np.ndarray): Координаты бокса [x1, y1, x2, y2].
        label_text (str): Текст метки класса.
        score (float): Уверенность модели (0.0 - 1.0).
        color (Tuple[int, int, int]): Цвет бокса (B, G, R). По умолчанию зеленый.
    """
    h_img, w_img, _ = img.shape
    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

    text = f"{label_text} {score:.2f}"
    font_scale = 0.4
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

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
        white_rect = np.full(roi.shape, 255, dtype=np.uint8)

        cv2.addWeighted(roi, 0.6, white_rect, 0.4, 1.0, dst=roi)
        img[ry1:ry2, rx1:rx2] = roi

    text_y = ry2 - 3 if ry1 < y1 else ry2 - 3
    cv2.putText(img, text, (x1 + 2, text_y), font, font_scale, (0, 0, 0), thickness)