# ==============================================================================
# test_dataset_setup.py — Script kiểm tra Custom Dataset & Vocab cho SeqTR
# ==============================================================================
import os
import json
import numpy as np
from PIL import Image
import torch

from utils.vocab import build_vocab, clean_expression
from datasets.dataset import CustomGroundingDataset, build_dataloader

def create_mock_data():
    """Tạo dữ liệu giả lập để kiểm tra."""
    print("1. Tạo dữ liệu giả lập...")
    
    # Tạo thư mục test_data tạm thời
    os.makedirs("test_data/images", exist_ok=True)
    
    # Tạo ảnh đen giả lập kích thước 100x100
    dummy_img = Image.new("RGB", (100, 100), color="black")
    dummy_img.save("test_data/images/004321.jpg")
    dummy_img.save("test_data/images/003704.jpg")
    
    # Cấu trúc JSON đúng như bạn đã mô tả
    mock_json = {
        "004321": {
            "filename": "004321.jpg",
            "width": 100,
            "height": 100,
            "bboxes": [
                {
                    "points": [
                        [10.0, 10.0],
                        [90.0, 10.0],
                        [90.0, 90.0],
                        [10.0, 90.0]
                    ],
                    "description": [
                        "Người đàn ông mặc đồ bảo hộ y tế đứng trước phòng bệnh."
                    ]
                }
            ]
        },
        "003704": {
            "filename": "003704.jpg",
            "width": 100,
            "height": 100,
            "bboxes": [
                {
                    "points": [
                        [20.0, 20.0],
                        [80.0, 20.0],
                        [80.0, 80.0],
                        [20.0, 80.0]
                    ],
                    "description": [
                        "nữ giáo viên mặc áo dài màu xanh"
                    ]
                }
            ]
        }
    }
    
    mock_ann_path = "test_data/mock_train.json"
    with open(mock_ann_path, "w", encoding="utf-8") as f:
        json.dump(mock_json, f, ensure_ascii=False, indent=2)
        
    return mock_ann_path, "test_data/images"

def main():
    try:
        mock_ann, mock_img_dir = create_mock_data()
        
        # 2. Kiểm tra phần build_vocab với pyvi
        print("\n2. Kiểm tra bộ từ vựng build_vocab (với pyvi)...")
        token2idx, idx2token = build_vocab(mock_ann)
        print(f"   -> Tổng số từ trong vocab: {len(token2idx)}")
        print(f"   -> Danh sách từ vựng: {list(token2idx.keys())}")
        
        # Kiểm tra xem từ ghép tiếng Việt đã được nối chính xác chưa
        expected_compound_words = ["người_đàn_ông", "bảo_hộ", "y_tế", "phòng_bệnh", "giáo_viên", "áo_dài"]
        found_compounds = [w for w in expected_compound_words if w in token2idx]
        print(f"   -> Tìm thấy từ ghép nối thành công: {found_compounds}")
        
        assert "áo_dài" in token2idx or "giáo_viên" in token2idx, "Lỗi: pyvi không tách được từ ghép chính xác!"
        
        # 3. Khởi tạo CustomGroundingDataset
        print("\n3. Kiểm tra khởi tạo CustomGroundingDataset...")
        dataset = CustomGroundingDataset(
            ann_file=mock_ann,
            img_dir=mock_img_dir,
            split="train",
            token2idx=token2idx,
            max_token=15,
            img_size=640
        )
        print(f"   -> Đọc thành công {len(dataset)} samples trong Dataset.")
        assert len(dataset) == 2, f"Lỗi: Số lượng sample không đúng, kỳ vọng 2 nhưng nhận {len(dataset)}"
        
        # 4. Kiểm tra lấy sample đầu tiên
        print("\n4. Kiểm tra lấy 1 sample ngẫu nhiên (Dataset[0])...")
        img, ref_inds, gt_bbox, img_meta = dataset[0]
        
        print(f"   -> Kích thước ảnh tensor đầu ra: {img.shape}") # Kỳ vọng: [3, 640, 640]
        print(f"   -> Tensor câu miêu tả (ref_inds): {ref_inds}")
        print(f"   -> Câu miêu tả sau khi giải mã ngược: {[idx2token[idx.item()] for idx in ref_inds if idx.item() > 0]}")
        print(f"   -> Bounding Box đầu ra (gt_bbox): {gt_bbox}")
        print(f"   -> Metadata của ảnh: {img_meta}")
        
        assert img.shape == (3, 640, 640), f"Lỗi: Kích thước ảnh sau padding phải là [3, 640, 640], nhận {img.shape}"
        assert gt_bbox.shape == (4,), f"Lỗi: Định dạng bbox phải có 4 tọa độ, nhận {gt_bbox.shape}"
        
        # 5. Kiểm tra DataLoader
        print("\n5. Kiểm tra DataLoader...")
        loader = build_dataloader(dataset, batch_size=2, shuffle=True, num_workers=0)
        batch = next(iter(loader))
        imgs, refs, bboxes, shapes = batch
        print(f"   -> Batch images shape: {imgs.shape}")
        print(f"   -> Batch refs shape: {refs.shape}")
        print(f"   -> Batch bboxes shape: {bboxes.shape}")
        print(f"   -> Batch shapes metadata: {shapes}")
        
        print("\n=======================================================")
        print("🎉 XIN CHÚC MỪNG! BẠN ĐÃ SỬA ĐÚNG CẢ 3 FILE DATASET & VOCAB!")
        print("=======================================================")
        
    except Exception as e:
        print(f"\n❌ KIỂM TRA THẤT BẠI: Đã xảy ra lỗi trong quá trình chạy thử:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
