# ==============================================================================
# transformer.py — Auto-Regressive Transformer + Sequence Head
# ==============================================================================
# Đây là MODULE CỐT LÕI — biến bài toán Visual Grounding thành bài toán
# sinh chuỗi (sequence generation), giống như dịch máy hay GPT sinh text.
#
# Ý tưởng:
#   Bounding box [x1, y1, x2, y2] = 4 số thực ∈ [0, 640]
#   → Quantize thành 4 tokens ∈ {0, 1, 2, ..., 999}
#   → Transformer Decoder sinh 4 tokens này TUẦN TỰ (auto-regressive):
#       Step 0: [START] → predict x1
#       Step 1: [START, x1] → predict y1
#       Step 2: [START, x1, y1] → predict x2
#       Step 3: [START, x1, y1, x2] → predict y2
#
# Cấu trúc:
#   ┌─────────────────────────┐
#   │  Input Projection       │  Conv1x1: 1024 → 256 channels
#   │  + 2D Positional Enc.   │  Sine encoding cho spatial positions
#   └──────────┬──────────────┘
#              ↓
#   ┌─────────────────────────┐
#   │  Transformer Encoder    │  6 layers, self-attention trên visual tokens
#   │  (6 layers)             │  [B, H*W, 256]
#   └──────────┬──────────────┘
#              ↓ memory
#   ┌─────────────────────────┐
#   │  Transformer Decoder    │  3 layers, cross-attention encoder↔decoder
#   │  (3 layers, causal)     │  Auto-regressive với causal mask
#   └──────────┬──────────────┘
#              ↓
#   ┌─────────────────────────┐
#   │  Predictor (3x FC)      │  256 → 256 → 1001 (num_bin + 1)
#   └──────────┬──────────────┘
#              ↓
#   logits [B, seq_len, 1001]  →  Cross-Entropy Loss
# ==============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# PHẦN 1: POSITIONAL ENCODING
# ==============================================================================

class SinePositionalEncoding2D(nn.Module):
    """
    Positional Encoding 2D dạng Sine/Cosine cho feature map ảnh.
    Output: [B, d_model, H, W] — mỗi pixel có 1 vector positional encoding.
    Sẽ được cộng vào các features map trước khi được flatten
    """

    def __init__(self, num_feature, temperature=10000, normalize=True):
        """
        Args:
            num_feature (int): d_model // 2. Vì encode X và Y riêng,
                mỗi chiều dùng num_feature dimensions → tổng = 2 * num_feature = d_model.
            temperature (int): Hằng số scaling. Giá trị lớn → tần số thấp hơn.
            normalize (bool): Có normalize tọa độ về [0, 2π] không.
        """
        super().__init__()
        self.num_feature = num_feature
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi  # Normalize tọa độ về [0, 2π]

    def forward(self, mask):
        """
        Args:
            mask (Tensor): [B, H, W], dtype=bool.
                True = vị trí bị pad (bỏ qua), False = vị trí hợp lệ.

        Returns:
            pos (Tensor): [B, d_model, H, W] — positional encoding cho mỗi pixel.
        """
        # not_mask: True ở vị trí hợp lệ, False ở vị trí pad
        not_mask = ~mask  # [B, H, W]

        # Tạo tọa độ y và x bằng cumulative sum
        # cumsum dọc theo axis 1 (H) → y tăng dần từ trên xuống: 1, 2, 3, ...
        # cumsum dọc theo axis 2 (W) → x tăng dần từ trái sang: 1, 2, 3, ...
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # [B, H, W]
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # [B, H, W]

        if self.normalize:
            # Normalize về [0, 2π]
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # Tạo vector tần số: [0, 1, 2, ..., num_feature-1]
        dim_t = torch.arange(self.num_feature, dtype=torch.float32, device=mask.device)
        # Công thức: temperature^(2i / num_feature)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feature)

        # Tính positional encoding cho x và y
        pos_x = x_embed[:, :, :, None] / dim_t  # [B, H, W, num_feature]
        pos_y = y_embed[:, :, :, None] / dim_t  # [B, H, W, num_feature]

        # Xen kẽ sin và cos: pos[..., 0::2] = sin, pos[..., 1::2] = cos
        B, H, W = mask.shape
        pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()], dim=4).view(B, H, W, -1)
        pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()], dim=4).view(B, H, W, -1)

        # Nối pos_y và pos_x → [B, H, W, d_model] → permute → [B, d_model, H, W]
        pos = torch.cat([pos_y, pos_x], dim=3).permute(0, 3, 1, 2)

        return pos


# ==============================================================================
# PHẦN 2: QUANTIZE / DEQUANTIZE
# ==============================================================================

def quantize_bbox(bbox, img_meta, num_bin=1000):
    """
    Chuyển bounding box từ tọa độ float sang tokens integer.

    Bounding box [x1, y1, x2, y2] nằm trong hệ tọa độ ảnh đã pad (640x640).
    Quantize = chia đều khoảng [0, pad_size] thành num_bin bins.

    Args:
        bbox (Tensor): [B, 4] — float, tọa độ [x1, y1, x2, y2]
        img_meta (list[dict]): Chứa 'pad_shape' = (640, 640, 3)
        num_bin (int): Số bins (1000)

    Returns:
        tokens (Tensor): [B, 4] — long, giá trị ∈ [0, 999]

    Ví dụ:
        bbox = [160.0, 80.0, 480.0, 400.0], pad_shape = (640, 640)
        norm = bbox / [640, 640, 640, 640] = [0.25, 0.125, 0.75, 0.625]
        tokens = norm * 1000 = [250, 125, 750, 625]
    """
    B = bbox.shape[0]

    # Lấy pad_shape (kích thước ảnh sau pad) cho mỗi sample
    # pad_shape = (H, W, 3) → lấy W, H
    pad_w = torch.tensor([m['pad_shape'][1] for m in img_meta],
                         device=bbox.device, dtype=bbox.dtype)  # [B]
    pad_h = torch.tensor([m['pad_shape'][0] for m in img_meta],
                         device=bbox.device, dtype=bbox.dtype)  # [B]

    # Tạo scale factor: [W, H, W, H] cho [x1, y1, x2, y2]
    scale = torch.stack([pad_w, pad_h, pad_w, pad_h], dim=1)  # [B, 4]

    # Normalize về [0, 1] rồi nhân với num_bin
    tokens = (bbox / scale * num_bin).long()

    # Clamp về [0, num_bin - 1] để tránh out-of-range
    tokens = tokens.clamp(0, num_bin - 1)

    return tokens


def dequantize_bbox(tokens, img_meta, num_bin=1000):
    """
    Chuyển tokens integer ngược lại thành tọa độ float.
    Dùng khi inference — model predict tokens → cần chuyển về bbox thật.

    Args:
        tokens (Tensor): [B, 4] — long, giá trị ∈ [0, 999]
        img_meta (list[dict]): Chứa 'pad_shape'
        num_bin (int): Số bins (1000)

    Returns:
        bbox (Tensor): [B, 4] — float, tọa độ [x1, y1, x2, y2]
    """
    B = tokens.shape[0]

    pad_w = torch.tensor([m['pad_shape'][1] for m in img_meta],
                         device=tokens.device, dtype=torch.float32)
    pad_h = torch.tensor([m['pad_shape'][0] for m in img_meta],
                         device=tokens.device, dtype=torch.float32)

    scale = torch.stack([pad_w, pad_h, pad_w, pad_h], dim=1)  # [B, 4]

    # tokens / num_bin → [0, 1] → * scale → tọa độ thật
    bbox = tokens.float() / num_bin * scale

    return bbox


# ==============================================================================
# PHẦN 3: SEQUENCE HEAD
# ==============================================================================

class SeqHead(nn.Module):
    """
    Sequence Head — phần cốt lõi của SeqTR.

    Nhận fused features [B, 1024, H, W] → sinh ra 4 tokens [x1, y1, x2, y2].
    Dùng auto-regressive Transformer decoder (giống GPT).
    """

    def __init__(self, in_ch=1024, d_model=256, nhead=8, dim_feedforward=1024,
                 dropout=0.1, enc_layers=6, dec_layers=3,
                 num_bin=1000, label_smoothing=0.1):
        """
        Args:
            in_ch (int): Input channels từ fusion (1024)
            d_model (int): Transformer dimension (256)
            nhead (int): Số attention heads (8)
            dim_feedforward (int): FFN hidden dim (1024)
            dropout (float): Dropout rate (0.1)
            enc_layers (int): Số Transformer Encoder layers (6)
            dec_layers (int): Số Transformer Decoder layers (3)
            num_bin (int): Số bins quantization (1000)
            label_smoothing (float): Label smoothing factor (0.1)
        """
        super().__init__()

        self.d_model = d_model
        self.num_bin = num_bin
        self.vocab_size = num_bin + 1  # 1001 (0-999 = tọa độ, 1000 = END token)
        self.seq_len = 4               # Bbox = 4 tokens

        # --- 1. Input Projection ---
        # Giảm channels: 1024 → 256 (d_model)
        # GroupNorm: ổn định hơn BatchNorm khi batch size nhỏ
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_ch, d_model, kernel_size=1, bias=True),
            nn.GroupNorm(32, d_model),  # 32 groups, mỗi group = 256/32 = 8 channels
        )

        # --- 2. Positional Encodings ---
        # 2D Sine encoding cho visual features (spatial position)
        self.pos_enc_2d = SinePositionalEncoding2D(
            num_feature=d_model // 2,  # 128 (sẽ nhân đôi cho x và y → 256)
            normalize=True
        )
        # 1D Learned encoding cho sequence tokens (temporal position)
        # num_embeddings = 5: [START, x1, y1, x2, y2] (5 vị trí)
        self.pos_enc_1d = nn.Embedding(
            num_embeddings=self.seq_len + 1,  # 5
            embedding_dim=d_model,             # 256
        )

        # --- 3. Token Embedding ---
        # Chuyển token index (0-1000) → vector d_model chiều
        # Dùng trong decoder: input của decoder là embedding của tokens đã predict
        self.token_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,  # 1001
            embedding_dim=d_model,           # 256
        )

        # --- 4. Transformer Encoder ---
        # Xử lý visual features bằng self-attention
        # Mỗi pixel "nhìn" toàn bộ feature map → hiểu ngữ cảnh toàn cục
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True,  # Input: [B, seq, d_model]
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)

        # --- 5. Transformer Decoder ---
        # Sinh tokens auto-regressive (tuần tự, mỗi token phụ thuộc token trước)
        # Cross-attention: decoder tokens attend vào encoder memory (visual features)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)

        # --- 6. Predictor ---
        # FC layers: d_model → d_model → vocab_size
        # Chuyển decoder output thành logits cho mỗi bin
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, self.vocab_size),  # → 1001 classes
        )

        # --- 7. Loss ---
        # Cross-Entropy Loss với label smoothing
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Khởi tạo weights
        self._reset_parameters()

    def _reset_parameters(self):
        """Khởi tạo weights bằng Xavier Uniform (giống paper gốc)."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_causal_mask(self, seq_len, device):
        """
        Tạo causal mask (tam giác trên) cho decoder.

        Causal mask đảm bảo token ở vị trí i chỉ attend vào các token ở vị trí ≤ i.
        Đây là cơ chế auto-regressive: khi predict y1, model chỉ "nhìn" x1,
        không được "nhìn trước" x2, y2.

        Ví dụ (seq_len=5):
            [0,   -inf, -inf, -inf, -inf]   ← START chỉ nhìn START
            [0,   0,    -inf, -inf, -inf]   ← x1 nhìn START, x1
            [0,   0,    0,    -inf, -inf]   ← y1 nhìn START, x1, y1
            [0,   0,    0,    0,    -inf]   ← x2 nhìn START, x1, y1, x2
            [0,   0,    0,    0,    0   ]   ← y2 nhìn tất cả

        Returns:
            mask (Tensor): [seq_len, seq_len], 0 = attend, -inf = block
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def _encode(self, x_fused, img_metas):
        """
        Encode visual features thành memory.

        Args:
            x_fused (Tensor): [B, 1024, H, W] — fused features
            img_metas (list[dict]): Metadata (dùng cho mask)

        Returns:
            memory (Tensor): [B, H*W, d_model] — encoded visual tokens
            x_mask (Tensor): [B, H*W] — padding mask
            x_pos (Tensor): [B, H*W, d_model] — positional encoding
        """
        B = x_fused.shape[0]

        # Project channels: 1024 → 256
        x = self.input_proj(x_fused)  # [B, 256, H, W]
        _, _, H, W = x.shape

        # Tạo mask cho padding regions — trực tiếp ở kích thước feature map
        # (Tiết kiệm memory: tạo [B, H, W] thay vì [B, 640, 640] rồi resize)
        # True ở vùng pad, False ở vùng ảnh thật
        input_h = img_metas[0]['pad_shape'][0]  # 640
        input_w = img_metas[0]['pad_shape'][1]  # 640
        x_mask = x_fused.new_ones((B, H, W))  # [B, H, W] toàn 1

        for i in range(B):
            img_h, img_w, _ = img_metas[i]['img_shape']
            # Scale img_shape xuống feature map size
            feat_h = int(round(img_h * H / input_h))
            feat_w = int(round(img_w * W / input_w))
            feat_h = min(feat_h, H)
            feat_w = min(feat_w, W)
            x_mask[i, :feat_h, :feat_w] = 0  # Vùng ảnh thật = 0

        x_mask = x_mask.bool()

        # Positional encoding 2D
        x_pos = self.pos_enc_2d(x_mask)  # [B, d_model, H, W]

        # Flatten spatial dims: [B, d_model, H, W] → [B, H*W, d_model]
        x = x.flatten(2).transpose(1, 2)          # [B, H*W, 256]
        x_pos = x_pos.flatten(2).transpose(1, 2)  # [B, H*W, 256]
        x_mask = x_mask.flatten(1)                  # [B, H*W]

        # Thêm positional encoding vào features
        x_with_pos = x + x_pos  # [B, H*W, 256]

        # Transformer Encoder: self-attention trên visual tokens
        memory = self.encoder(
            x_with_pos,
            src_key_padding_mask=x_mask  # Ignore padding positions
        )  # [B, H*W, 256]

        return memory, x_mask, x_pos

    def forward_train(self, x_fused, gt_bbox, img_metas):
        """
        Forward pass khi TRAINING — dùng Teacher Forcing.

        Teacher Forcing: thay vì dùng output predict của model làm input step tiếp,
        ta dùng GROUND TRUTH. Giúp training ổn định hơn.

        Ví dụ: GT bbox tokens = [250, 125, 750, 625]

        Input cho decoder (shift right):
            [START_EMB, embed(250), embed(125), embed(750), embed(625)]
            → positions:  0          1           2           3          4

        Target (labels):
            [250, 125, 750, 625, END]
            → model phải predict đúng token tại mỗi position

        Args:
            x_fused (Tensor): [B, 1024, H, W]
            gt_bbox (Tensor): [B, 4] — ground truth bbox [x1, y1, x2, y2]
            img_metas (list[dict])

        Returns:
            loss (Tensor): Scalar — cross-entropy loss
        """
        B = x_fused.shape[0]
        device = x_fused.device

        # 1. Encode visual features → memory
        memory, x_mask, x_pos = self._encode(x_fused, img_metas)

        # 2. Quantize GT bbox → tokens
        # [160.0, 80.0, 480.0, 400.0] → [250, 125, 750, 625]
        gt_tokens = quantize_bbox(gt_bbox, img_metas, self.num_bin)  # [B, 4]

        # 3. Tạo target (thêm END token ở cuối)
        end_token = torch.full((B, 1), self.num_bin, dtype=torch.long, device=device)
        targets = torch.cat([gt_tokens, end_token], dim=1)  # [B, 5]

        # 4. Tạo decoder input (Teacher Forcing — shift right)
        # START embedding = vector 0 (learned sẽ tự điều chỉnh qua training)
        start_embed = torch.zeros(B, 1, self.d_model, device=device)  # [B, 1, 256]
        gt_embeds = self.token_embedding(gt_tokens)                    # [B, 4, 256]
        seq_input = torch.cat([start_embed, gt_embeds], dim=1)         # [B, 5, 256]

        # 5. Thêm positional encoding 1D cho sequence
        seq_pos = self.pos_enc_1d(
            torch.arange(self.seq_len + 1, device=device)  # [0, 1, 2, 3, 4]
        ).unsqueeze(0).expand(B, -1, -1)  # [B, 5, 256]
        seq_input = seq_input + seq_pos

        # 6. Tạo causal mask
        causal_mask = self._generate_causal_mask(self.seq_len + 1, device)  # [5, 5]

        # 7. Decode
        decoder_out = self.decoder(
            seq_input,                          # tgt: [B, 5, 256]
            memory,                             # memory: [B, H*W, 256]
            tgt_mask=causal_mask,               # Causal mask [5, 5]
            memory_key_padding_mask=x_mask,     # Ignore padded visual tokens
        )  # [B, 5, 256]

        # 8. Predict logits
        logits = self.predictor(decoder_out)  # [B, 5, 1001]

        # 9. Compute loss
        # Reshape cho CrossEntropyLoss: logits [B*5, 1001], targets [B*5]
        loss = self.loss_fn(
            logits.reshape(-1, self.vocab_size),  # [B*5, 1001]
            targets.reshape(-1)                    # [B*5]
        )

        return loss

    @torch.no_grad()
    def forward_test(self, x_fused, img_metas):
        """
        Forward pass khi INFERENCE — sinh tokens auto-regressive.

        Khác với training (dùng GT làm input), inference phải tự sinh từng token
        rồi dùng token đó làm input cho step tiếp theo.

        Step 0: [START]                        → predict x1
        Step 1: [START, embed(x1)]             → predict y1
        Step 2: [START, embed(x1), embed(y1)]  → predict x2
        Step 3: [START, embed(x1), embed(y1), embed(x2)] → predict y2

        Args:
            x_fused (Tensor): [B, 1024, H, W]
            img_metas (list[dict])

        Returns:
            pred_bbox (Tensor): [B, 4] — predicted bbox [x1, y1, x2, y2] float
        """
        B = x_fused.shape[0]
        device = x_fused.device

        # 1. Encode
        memory, x_mask, x_pos = self._encode(x_fused, img_metas)

        # 2. Auto-regressive generation
        # Khởi tạo sequence với START token
        start_embed = torch.zeros(B, 1, self.d_model, device=device)
        seq_input = start_embed  # [B, 1, 256]
        output_tokens = []

        for step in range(self.seq_len):  # 4 steps
            # Thêm positional encoding
            cur_len = seq_input.shape[1]
            seq_pos = self.pos_enc_1d(
                torch.arange(cur_len, device=device)
            ).unsqueeze(0).expand(B, -1, -1)
            seq_with_pos = seq_input + seq_pos

            # Causal mask
            causal_mask = self._generate_causal_mask(cur_len, device)

            # Decode
            decoder_out = self.decoder(
                seq_with_pos,
                memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=x_mask,
            )

            # Predict token tại vị trí cuối
            logits = self.predictor(decoder_out[:, -1, :])  # [B, 1001]
            next_token = logits.argmax(dim=-1)               # [B] — greedy decoding

            output_tokens.append(next_token)

            # Thêm token mới vào sequence cho step tiếp theo
            next_embed = self.token_embedding(next_token).unsqueeze(1)  # [B, 1, 256]
            seq_input = torch.cat([seq_input, next_embed], dim=1)

        # 3. Stack tokens và dequantize
        pred_tokens = torch.stack(output_tokens, dim=1)  # [B, 4]
        pred_bbox = dequantize_bbox(pred_tokens, img_metas, self.num_bin)

        return pred_bbox


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":
    print("=== Test SeqHead ===")

    head = SeqHead(
        in_ch=1024, d_model=256, nhead=8, dim_feedforward=1024,
        dropout=0.1, enc_layers=6, dec_layers=3,
        num_bin=1000, label_smoothing=0.1
    )

    total = sum(p.numel() for p in head.parameters())
    print(f"Params: {total:,}")

    # Giả lập input
    B = 2
    x_fused = torch.randn(B, 1024, 40, 40)
    gt_bbox = torch.tensor([[100.0, 50.0, 400.0, 300.0],
                             [200.0, 100.0, 500.0, 400.0]])
    img_metas = [
        {'pad_shape': (640, 640, 3), 'img_shape': (480, 640, 3)},
        {'pad_shape': (640, 640, 3), 'img_shape': (640, 480, 3)},
    ]

    # Test training
    print("\n--- Test forward_train ---")
    loss = head.forward_train(x_fused, gt_bbox, img_metas)
    print(f"Loss: {loss.item():.4f}")

    # Test inference
    print("\n--- Test forward_test ---")
    pred_bbox = head.forward_test(x_fused, img_metas)
    print(f"Predicted bbox: {pred_bbox}")
    print(f"Shape: {pred_bbox.shape}")  # [2, 4]

    print("\n✅ SeqHead test passed!")
