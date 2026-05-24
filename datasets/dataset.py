import os
import re
import json
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from utils.vocab import clean_expression, tokenize_expression, build_vocab



def resize_image_keep_ratio(img, max_size):
    """
    Resize ảnh sao cho cạnh dài nhất = max_size, GIỮ NGUYÊN tỉ lệ.

    Args:
        img (np.ndarray): Ảnh gốc, shape [H, W, 3], dtype=uint8
        max_size (int): Kích thước tối đa (640)

    Returns:
        resized_img (np.ndarray): Ảnh đã resize, shape [new_H, new_W, 3]
        scale (float): Tỉ lệ scale. Ví dụ: 0.8 nghĩa là ảnh bị thu nhỏ 80%

    Ví dụ:
        Ảnh 800x600, max_size=640
        scale = 640 / max(800, 600) = 640/800 = 0.8
        new_size = (800*0.8, 600*0.8) = (640, 480)
    """

    h,w = img.shape[:2]

    scale = max_size / max(h,w)

    new_h , new_w = int(h*scale), int(w*scale)

    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((new_w, new_h))

    resized_img = np.array(pil_img)

    return resized_img, scale

def pad_image_to_square(img, target_size, pad_value =0):
     
        """
        Pad ảnh về kích thước target_size x target_size.
        Padding thêm ở bên PHẢI và bên DƯỚI.

        Args:
            img (np.ndarray): Ảnh đã resize, shape [H, W, 3]
            target_size (int): Kích thước đích (640)
            pad_value (int): Giá trị pixel để pad (0 = đen)

        Returns:
            padded_img (np.ndarray): Ảnh đã pad, shape [target_size, target_size, 3]

        Ví dụ:
            Ảnh 640x480 → pad thêm 160 dòng phía dưới → 640x640
            ┌──────────────┐
            │  Ảnh gốc     │ 480px
            │  640 x 480   │
            ├──────────────┤
            │  Padding (0) │ 160px
            └──────────────┘
                640px
        """
        h,w = img.shape[:2]

        padded = np.full((target_size,target_size,3), pad_value, dtype = img.dtype)

        padded[:h, :w, :] = img
        
        return padded

def normalize_image(img):
    """
    Chuyển các giá trị pixel về khoảng [0, 1]

    """

    return img.astype(np.float32)/255.0

def image_to_tensor(img):
    """
    Chuyển numpy image thành tensor, đổi thứ tự axes từ [H, W, C] → [C, H, W]
    """

    img = np.transpose(img, (2, 0, 1))
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)

    return img

def transform_bbox(bbox_xywh, scale, img_shape_after_resize):
    """
    Chuyển đổi bounding box cho phù hợp với ảnh đã resize.

    Annotations gốc lưu bbox dạng [x, y, w, h] (COCO format) -> chuyển sang [x1, y1, x2, y2] (corner format) rồi scale.

    Args:
        bbox_xywh (list): [x, y, w, h] từ annotation gốc
        scale (float): Tỉ lệ resize đã áp dụng lên ảnh
        img_shape_after_resize (tuple): (new_h, new_w) sau khi resize

    Returns:
        bbox (Tensor): [4], dạng [x1, y1, x2, y2], dtype=torch.float32

    Ví dụ:
        bbox_xywh = [100, 50, 200, 150]  → gốc: top-left (100,50), kích thước 200x150
        scale = 0.8
        → Sau scale: [80, 40, 160, 120]  → xywh format
        → Chuyển sang xyxy: [80, 40, 240, 160]  → x2 = x1 + w, y2 = y1 + h
        → Clip: đảm bảo tọa độ không vượt quá kích thước ảnh
    """
    x, y, w, h = bbox_xywh

    # Scale tọa độ theo tỉ lệ resize
    x1 = x * scale
    y1 = y * scale
    x2 = (x + w) * scale
    y2 = (y + h) * scale

    # Clip tọa độ để không vượt quá biên ảnh
    new_h, new_w = img_shape_after_resize
    x1 = np.clip(x1, 0, new_w - 1)
    y1 = np.clip(y1, 0, new_h - 1)
    x2 = np.clip(x2, 0, new_w - 1)
    y2 = np.clip(y2, 0, new_h - 1)

    return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

class CustomGroundingDataset(Dataset):
    """
    Dataset class cho bộ dữ liệu Visual Grounding tùy chỉnh tiếng Việt (Bản Chống Lỗi).
    """
    def __init__(self, ann_file, img_dir, split, token2idx, max_token=15, img_size=640):
        super().__init__()
        self.img_dir = img_dir
        self.split = split
        self.token2idx = token2idx
        self.max_token = max_token
        self.img_size = img_size
        # Đọc dữ liệu JSON tùy chỉnh
        with open(ann_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.anns = []
        for img_id, img_info in data.items():
            # Bỏ qua nếu dữ liệu không chuẩn
            if not isinstance(img_info, dict) or 'filename' not in img_info:
                continue
                
            filename = img_info['filename']
            bboxes = img_info.get('bboxes', [])
            
            if bboxes is None:
                bboxes = []
            for bbox_idx, bbox_info in enumerate(bboxes):
                # 1. TỰ ĐỘNG BỎ QUA NẾU THIẾU 'points'
                if not isinstance(bbox_info, dict) or 'points' not in bbox_info:
                    print(f"⚠️ Bỏ qua 1 bbox của ảnh '{img_id}' do thiếu 'points' hoặc sai cấu trúc.")
                    continue
                
                # 2. Bỏ qua nếu thiếu description
                expressions = bbox_info.get('description', [])
                if not expressions:
                    continue
                try:
                    points = bbox_info['points']
                    
                    # Trích xuất xmin, ymin, xmax, ymax từ list 4 points
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    xmin, xmax = min(xs), max(xs)
                    ymin, ymax = min(ys), max(ys)
                    
                    w = xmax - xmin
                    h = ymax - ymin
                    bbox_xywh = [xmin, ymin, w, h]
                    self.anns.append({
                        'image_id': img_id,
                        'filename': filename,
                        'bbox': bbox_xywh,
                        'expressions': expressions
                    })
                except Exception as e:
                    print(f"⚠️ Lỗi khi xử lý tọa độ ở ảnh '{img_id}', bỏ qua: {e}")
                    continue
        print(f"[{split}] Loaded {len(self.anns)} valid samples từ {os.path.basename(ann_file)}")
    def __len__(self):
        return len(self.anns)
    def __getitem__(self, index):
        ann = self.anns[index]
        # Nạp ảnh trực tiếp bằng filename
        img_path = os.path.join(self.img_dir, ann['filename'])
        pil_img = Image.open(img_path).convert('RGB')
        img = np.array(pil_img)
        ori_h, ori_w = img.shape[:2]
        # Tiền xử lý ảnh (giữ nguyên logic gốc của SeqTR)
        img, scale = resize_image_keep_ratio(img, self.img_size)
        resized_h, resized_w = img.shape[:2]
        img = pad_image_to_square(img, self.img_size)
        img = normalize_image(img)      # [H, W, 3] float32 [0, 1]
        img = image_to_tensor(img)      # [3, H, W] tensor
        expressions = ann['expressions']
        if self.split == 'train':
            # Chọn ngẫu nhiên câu mô tả khi train
            expression = random.choice(expressions)
        else:
            expression = expressions[0]
        # Dùng tokenize_expression đã cập nhật tách từ tiếng Việt
        ref_inds = tokenize_expression(expression, self.token2idx, self.max_token)
        # Scale bbox
        gt_bbox = transform_bbox(
            ann['bbox'],
            scale=scale,
            img_shape_after_resize=(resized_h, resized_w)
        )
        img_meta = {
            'image_id': ann['image_id'],
            'expression': expression,
            'ori_shape': (ori_h, ori_w, 3),
            'img_shape': (resized_h, resized_w, 3),
            'pad_shape': (self.img_size, self.img_size, 3),
            'scale_factor': np.array([scale, scale, scale, scale], dtype=np.float32),
        }
        return img, ref_inds, gt_bbox, img_meta


def collate_fn(batch):
    """
    Custom collate function cho DataLoader.

    DataLoader mặc định chỉ gom được tensors cùng shape.
    img_meta là dict nên cần xử lý riêng.
    """
    imgs, ref_inds, gt_bboxes, img_metas = zip(*batch)

    imgs = torch.stack(imgs, dim=0)
    ref_inds = torch.stack(ref_inds, dim=0)
    gt_bboxes = torch.stack(gt_bboxes, dim=0)

    # [CŨ] img_metas giữ nguyên dạng list of dicts
    # img_metas = list(img_metas)
    # return imgs, ref_inds, gt_bboxes, img_metas

    # [MỚI] Chuyển img_metas → tensor [B, 4] để DataParallel chia được
    # Mỗi dòng = [pad_h, pad_w, img_h, img_w]
    img_shapes = torch.tensor([
        [m['pad_shape'][0], m['pad_shape'][1],
         m['img_shape'][0], m['img_shape'][1]]
        for m in img_metas
    ], dtype=torch.float32)

    return imgs, ref_inds, gt_bboxes, img_shapes

def build_dataloader(dataset, batch_size, shuffle=True, num_workers=2):
    """
    Tạo DataLoader từ dataset.  

    Args:
        dataset: RefCOCODataset instance
        batch_size (int): Số sample mỗi batch
        shuffle (bool): Có xáo trộn dữ liệu không (True cho train, False cho val)
        num_workers (int): Số process song song load data

    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,      # Dùng custom collate vì có img_meta
        pin_memory=True,             # Tăng tốc CPU→GPU transfer
        drop_last=(shuffle == True), # Drop batch cuối nếu không đủ (chỉ khi train)
    )

if __name__ == "__main__":
    """
    Để test xem dataset hoạt động đúng chưa.
    Cần có file instances.json và ảnh COCO ở đúng đường dẫn trong config.
    """
    from config import Config

    print("=" * 60)
    print("TEST DATASET")
    print("=" * 60)

    # 1. Build vocabulary
    print("\n--- Building vocabulary ---")
    token2idx, idx2token = build_vocab(Config.ann_file)
    print(f"Vocabulary size: {len(token2idx)}")
    print(f"Sample words: {list(token2idx.items())[:10]}")

    # 2. Tạo dataset
    print("\n--- Creating dataset ---")
    train_dataset = CustomGroundingDataset(
        ann_file=Config.ann_file,
        img_dir=Config.img_dir,
        split='train',
        token2idx=token2idx,
        max_token=Config.max_token,
        img_size=Config.img_size,
    )

    # 3. Lấy 1 sample
    print("\n--- Getting 1 sample ---")
    img, ref_inds, gt_bbox, img_meta = train_dataset[0]
    print(f"Image shape: {img.shape}")            # [3, 640, 640]
    print(f"Image dtype: {img.dtype}")             # torch.float32
    print(f"Image range: [{img.min():.2f}, {img.max():.2f}]")  # [0, 1]
    print(f"Ref indices: {ref_inds}")              # [idx1, idx2, ..., 0, 0, 0]
    print(f"Ref words: {[idx2token.get(i.item(), '?') for i in ref_inds if i > 0]}")
    print(f"GT bbox: {gt_bbox}")                   # [x1, y1, x2, y2]
    print(f"Image meta: {img_meta}")

    # 4. Test DataLoader
    print("\n--- Testing DataLoader ---")
    loader = build_dataloader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    imgs, refs, bboxes, metas = batch
    print(f"Batch images: {imgs.shape}")    # [4, 3, 640, 640]
    print(f"Batch refs: {refs.shape}")      # [4, 15]
    print(f"Batch bboxes: {bboxes.shape}")  # [4, 4]
    print(f"Batch metas: {len(metas)} dicts")  # 4
