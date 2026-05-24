# ==============================================================================
# evaluate.py — Đánh giá model (Validation / Test)
# ==============================================================================
# Tính metric Accuracy@IoU>=0.5:
#   Với mỗi sample, nếu IoU giữa predicted bbox và ground truth bbox >= 0.5
#   → đúng (correct). Accuracy = số đúng / tổng số samples.
#
# Đây là metric chuẩn của bài toán Referring Expression Comprehension (REC).
# ==============================================================================

import torch
from torch.utils.data import DataLoader


def compute_iou_batch(pred, gt):
    """
    Tính IoU (Intersection over Union) cho 1 batch bounding boxes.

    IoU = diện tích giao / diện tích hợp
    Giá trị ∈ [0, 1]. IoU = 1 nghĩa là 2 bbox trùng khớp hoàn toàn.

    Args:
        pred (Tensor): [B, 4] — predicted bbox [x1, y1, x2, y2]
        gt (Tensor): [B, 4] — ground truth bbox [x1, y1, x2, y2]

    Returns:
        iou (Tensor): [B] — IoU cho từng sample trong batch
    """
    # Tọa độ vùng giao (intersection)
    # Góc trên-trái của intersection = max(pred_topleft, gt_topleft)
    # Góc dưới-phải của intersection = min(pred_bottomright, gt_bottomright)
    inter_x1 = torch.max(pred[:, 0], gt[:, 0])
    inter_y1 = torch.max(pred[:, 1], gt[:, 1])
    inter_x2 = torch.min(pred[:, 2], gt[:, 2])
    inter_y2 = torch.min(pred[:, 3], gt[:, 3])

    # Diện tích intersection (clamp 0 nếu không giao nhau)
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Diện tích mỗi bbox
    pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    gt_area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])

    # Union = pred + gt - intersection
    union_area = pred_area + gt_area - inter_area

    # IoU (thêm eps tránh chia 0)
    iou = inter_area / (union_area + 1e-6)

    return iou


@torch.no_grad()
def evaluate(model, dataloader, device, desc="Evaluating"):
    """
    Đánh giá model trên 1 split (val, testA, hoặc testB).

    Args:
        model: SeqTRDet model
        dataloader: DataLoader cho split cần đánh giá
        device: 'cuda' hoặc 'cpu'
        desc (str): Tên split (dùng để in log)

    Returns:
        accuracy (float): Accuracy@IoU>=0.5 (%)
        avg_iou (float): IoU trung bình (%)
    """
    model.eval()

    total_correct = 0
    total_samples = 0
    total_iou = 0.0

    for batch_idx, (imgs, ref_inds, gt_bboxes, img_shapes) in enumerate(dataloader):
        imgs = imgs.to(device)
        ref_inds = ref_inds.to(device)
        gt_bboxes = gt_bboxes.to(device)
        img_shapes = img_shapes.to(device)  # [MỚI]

        # Forward inference (không truyền gt_bbox → model trả về predicted bbox)
        # [CŨ] pred_bboxes = model(imgs, ref_inds, img_metas, gt_bbox=None)
        pred_bboxes = model(imgs, ref_inds, img_shapes, gt_bbox=None)

        # Tính IoU
        iou = compute_iou_batch(pred_bboxes, gt_bboxes)  # [B]

        # Đếm số sample có IoU >= 0.5
        correct = (iou >= 0.5).sum().item()
        total_correct += correct
        total_samples += imgs.shape[0]
        total_iou += iou.sum().item()

    accuracy = total_correct / total_samples * 100
    avg_iou = total_iou / total_samples * 100

    print(f"[{desc}] Accuracy@IoU>=0.5: {accuracy:.2f}% | "
          f"Avg IoU: {avg_iou:.2f}% | "
          f"Samples: {total_samples}")

    return accuracy, avg_iou


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":
    print("=== Test evaluate functions ===")

    # Test IoU
    pred = torch.tensor([[0, 0, 100, 100],
                          [0, 0, 50, 50]], dtype=torch.float32)
    gt = torch.tensor([[0, 0, 100, 100],
                        [25, 25, 75, 75]], dtype=torch.float32)

    iou = compute_iou_batch(pred, gt)
    print(f"IoU: {iou}")
    # Expected: [1.0, ...] (case 1 hoàn toàn trùng)

    assert abs(iou[0].item() - 1.0) < 1e-5, "IoU case 1 sai!"
    print("✅ evaluate test passed!")
