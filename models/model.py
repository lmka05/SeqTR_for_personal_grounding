# ==============================================================================
# model.py — Model tổng hợp SeqTR Detection
# ==============================================================================
# File này ghép 4 module lại thành 1 model hoàn chỉnh:
#
#   ┌─────────────┐    ┌─────────────┐
#   │  Ảnh (img)  │    │  Câu (text) │
#   └──────┬──────┘    └──────┬──────┘
#          ↓                  ↓
#   ┌─────────────┐    ┌─────────────┐
#   │  Backbone   │    │  Language   │
#   │ (ResNet-50) │    │  (BiGRU)   │
#   └──────┬──────┘    └──────┬──────┘
#          ↓                  ↓
#          └────────┬─────────┘
#                   ↓
#          ┌─────────────┐
#          │   Fusion    │
#          │ (tanh gate) │
#          └──────┬──────┘
#                 ↓
#          ┌─────────────┐
#          │  SeqHead    │
#          │(Transformer)│
#          └──────┬──────┘
#                 ↓
#          [x1, y1, x2, y2]
# ==============================================================================

import torch
import torch.nn as nn

from .backbone import VisualEncoder
from .language import LanguageEncoder
from .fusion import SimpleFusion
from .transformer import SeqHead


class SeqTRDet(nn.Module):
    """
    SeqTR Detection — Model hoàn chỉnh cho Visual Grounding (bounding box).

    Input:
        - img: [B, 3, 640, 640] — ảnh
        - ref_inds: [B, max_token] — câu đã tokenize
        - img_metas: list[dict] — metadata
        - gt_bbox: [B, 4] (chỉ khi train)

    Output:
        - Training: loss (scalar)
        - Inference: pred_bbox [B, 4]
    """

    def __init__(self, config, glove_vectors):
        """
        Args:
            config: Config object chứa hyperparameters
            glove_vectors (Tensor): [vocab_size, 300] — ma trận GloVe embedding
        """
        super().__init__()

        # 1. Visual Encoder — trích xuất features ảnh
        self.vis_enc = VisualEncoder(freeze_layers=True)

        # 2. Language Encoder — mã hóa câu mô tả
        self.lan_enc = LanguageEncoder(
            glove_vectors=glove_vectors,
            hidden_size=config.gru_hidden,  # 512 → output 1024
        )

        # 3. Fusion — kết hợp visual + language
        self.fusion = SimpleFusion(
            vis_channels=[512, 1024, 2048]  # Channels từ ResNet layer2, 3, 4
        )

        # 4. Sequence Head — Transformer auto-regressive decoder
        self.head = SeqHead(
            in_ch=config.backbone_out_channels,  # 1024
            d_model=config.d_model,               # 256
            nhead=config.nhead,                    # 8
            dim_feedforward=config.dim_feedforward, # 1024
            dropout=config.dropout,                # 0.1
            enc_layers=config.enc_layers,          # 6
            dec_layers=config.dec_layers,          # 3
            num_bin=config.num_bin,                 # 1000
            label_smoothing=config.label_smoothing, # 0.1
        )

    def forward(self, img, ref_inds, img_shapes, gt_bbox=None):
        """
        Forward pass — tự động switch giữa train/test dựa vào gt_bbox.

        Args:
            img (Tensor): [B, 3, 640, 640] — batch ảnh
            ref_inds (Tensor): [B, max_token] — batch câu đã tokenize
            # [CŨ] img_metas (list[dict]): Metadata cho mỗi ảnh
            img_shapes (Tensor): [B, 4] — [pad_h, pad_w, img_h, img_w] (tensor để DataParallel chia được)
            gt_bbox (Tensor | None): [B, 4] — GT bbox (None khi inference)

        Returns:
            Training: loss (Tensor scalar)
            Inference: pred_bbox (Tensor [B, 4])
        """
        # [MỚI] Tạo lại img_metas (list of dicts) từ tensor img_shapes
        # Vì transformer.py vẫn cần format dict
        B = img.shape[0]
        img_metas = []
        for i in range(B):
            img_metas.append({
                'pad_shape': (int(img_shapes[i, 0]), int(img_shapes[i, 1]), 3),
                'img_shape': (int(img_shapes[i, 2]), int(img_shapes[i, 3]), 3),
            })

        # Bước 1: Trích xuất visual features — 3 feature maps
        vis_feats = self.vis_enc(img)  # [C3, C4, C5]

        # Bước 2: Mã hóa câu mô tả → 1 vector
        lang_feat = self.lan_enc(ref_inds)  # [B, 1, 1024]

        # Bước 3: Kết hợp visual + language
        x_fused = self.fusion(vis_feats, lang_feat)  # [B, 1024, 20, 20]

        # Bước 4: Sinh tọa độ bbox
        if gt_bbox is not None:
            # TRAINING: tính loss (teacher forcing)
            loss = self.head.forward_train(x_fused, gt_bbox, img_metas)
            return loss
        else:
            # INFERENCE: sinh bbox auto-regressive
            pred_bbox = self.head.forward_test(x_fused, img_metas)
            return pred_bbox


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from config import Config

    print("=== Test SeqTRDet (Full Model) ===")

    # Tạo GloVe giả
    vocab_size = 100
    fake_glove = torch.randn(vocab_size, Config.glove_dim)
    fake_glove[0] = 0  # PAD = zero vector

    # Build model
    model = SeqTRDet(Config, fake_glove)

    # Đếm parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Frozen params:    {frozen:,}")

    # Giả lập input
    B = 2
    img = torch.randn(B, 3, 640, 640)
    ref_inds = torch.randint(1, vocab_size, (B, Config.max_token))
    gt_bbox = torch.tensor([[100.0, 50.0, 400.0, 300.0],
                             [200.0, 100.0, 500.0, 400.0]])
    img_metas = [
        {'pad_shape': (640, 640, 3), 'img_shape': (480, 640, 3)},
        {'pad_shape': (640, 640, 3), 'img_shape': (640, 480, 3)},
    ]

    # --- Test Training ---
    print("\n--- Training mode ---")
    model.train()
    loss = model(img, ref_inds, img_metas, gt_bbox=gt_bbox)
    print(f"Loss: {loss.item():.4f}")

    # Kiểm tra gradient flow
    loss.backward()
    grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"Parameters with gradient: {grad_count}")

    # --- Test Inference ---
    print("\n--- Inference mode ---")
    model.eval()
    pred_bbox = model(img, ref_inds, img_metas, gt_bbox=None)
    print(f"Predicted bbox: {pred_bbox}")
    print(f"Shape: {pred_bbox.shape}")  # Expected: [2, 4]

    print("\n🎉 SeqTRDet full model test passed!")
