"""
Visualize bounding boxes and descriptions on images.
Usage: python visualize_bbox.py
"""

import collections
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import textwrap

# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN - CHỈNH SỬA TẠI ĐÂY
# ============================================================
ANNOTATION_FILE = r"D:/Study_UIT/Personal Grounding/data/train.json"   # Đường dẫn tới file annotation JSON
IMAGE_DIR = r"D:/Study_UIT/Personal Grounding/data/images/images"             # Đường dẫn tới thư mục chứa ảnh
IMAGE_ID = "003704"                   # ID của ảnh cần hiển thị (key trong JSON)
# ============================================================

# Bảng màu đẹp cho các bounding box khác nhau
COLORS = [
    "#FF6B6B",  # Đỏ san hô
    "#4ECDC4",  # Xanh ngọc
    "#FFE66D",  # Vàng chanh
    "#A78BFA",  # Tím lavender
    "#F97316",  # Cam
    "#06B6D4",  # Xanh cyan
    "#EC4899",  # Hồng
    "#84CC16",  # Xanh lá
]


def load_annotations(annotation_file):
    """Đọc file annotation JSON."""
    with open(annotation_file, "r", encoding="utf-8") as f:
        return json.load(f)


def visualize(annotation_file, image_dir, image_id):
    """
    Hiển thị ảnh với bounding boxes và mô tả.

    Args:
        annotation_file: Đường dẫn tới file JSON chứa annotation.
        image_dir: Đường dẫn tới thư mục chứa ảnh.
        image_id: ID (key) của ảnh trong file JSON.
    """
    # Đọc annotations
    annotations = load_annotations(annotation_file)
    i = 0
    for i,ann in enumerate(annotations):
        print(annotations[ann])
        i+=1
        if i >=3:
            break
    if image_id not in annotations:
        print(f"[LỖI] Không tìm thấy image_id '{image_id}' trong file annotation.")
        print(f"Các ID có sẵn: {list(annotations.keys())[:10]}...")
        return

    entry = annotations[image_id]
    filename = entry["filename"]
    bboxes = entry["bboxes"]
    img_path = os.path.join(image_dir, filename)

    if not os.path.exists(img_path):
        print(f"[LỖI] Không tìm thấy ảnh: {img_path}")
        return

    # Đọc ảnh
    img = Image.open(img_path).convert("RGB")

    # Tạo figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img)
    ax.set_title(f"Image: {filename}  |  {len(bboxes)} bounding box(es)", fontsize=14, fontweight="bold")
    ax.axis("off")

    # Vẽ từng bounding box
    legend_texts = []
    for idx, bbox_info in enumerate(bboxes):
        points = bbox_info["points"]
        descriptions = bbox_info["description"]
        color = COLORS[idx % len(COLORS)]

        # Lấy tọa độ từ 4 điểm (top-left, top-right, bottom-right, bottom-left)
        x_min = min(p[0] for p in points)
        y_min = min(p[1] for p in points)
        x_max = max(p[0] for p in points)
        y_max = max(p[1] for p in points)
        width = x_max - x_min
        height = y_max - y_min

        # Vẽ hình chữ nhật
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2.5,
            edgecolor=color,
            facecolor=color,
            alpha=0.15,
        )
        ax.add_patch(rect)

        # Vẽ viền đậm hơn
        rect_border = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2.5,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect_border)

        # Gắn nhãn số thứ tự lên góc trên bên trái của bbox
        label = f"[{idx + 1}]"
        ax.text(
            x_min, y_min - 4,
            label,
            fontsize=12,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, edgecolor="none", alpha=0.9),
        )

        # Chuẩn bị mô tả cho legend
        desc_text = " | ".join(descriptions)
        wrapped = textwrap.fill(desc_text, width=60)
        legend_texts.append(f"[{idx + 1}] {wrapped}")

    # Hiển thị mô tả bên dưới ảnh
    desc_block = "\n".join(legend_texts)
    fig.text(
        0.5, 0.01,
        desc_block,
        ha="center", va="bottom",
        fontsize=10,
        family="sans-serif",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F0F0", edgecolor="#CCCCCC", alpha=0.95),
        wrap=True,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08 + 0.04 * len(bboxes))  # Chừa chỗ cho mô tả
    plt.show()


if __name__ == "__main__":
    visualize(ANNOTATION_FILE, IMAGE_DIR, IMAGE_ID)
