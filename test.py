import os
import sys
import argparse

import torch
from gensim.models import KeyedVectors

from config import Config
from utils.vocab import build_vocab, build_w2v_matrix
from datasets.dataset import CustomGroundingDataset, build_dataloader
from models.model import SeqTRDet
from evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description='SeqTR Detection — Test')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Đường dẫn tới file checkpoint (.pth)')
    parser.add_argument('--splits', nargs='+', default=['val', 'testA', 'testB'],
                        help='Các split cần đánh giá (mặc định: val testA testB)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size cho evaluation (mặc định: 64)')
    args = parser.parse_args()

    config = Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Splits: {args.splits}")

    # 1. Build vocab + GloVe
    print("\n--- Building vocab ---")
    token2idx, idx2token = build_vocab([config.train_ann_file, config.dev_ann_file])
    print(f"Vocab size: {len(token2idx)}")

    if os.path.exists(config.w2v_path):
        print(f"Loading Word2Vec model from {config.w2v_path}...")
        w2v_model = KeyedVectors.load_word2vec_format(
            config.w2v_path, 
            binary=config.w2v_is_binary
        )
        glove_matrix = build_w2v_matrix(token2idx, w2v_model, config.glove_dim)
        del w2v_model
        import gc; gc.collect()
    else:
        print(f"⚠️ Không tìm thấy file Word2Vec tại {config.w2v_path}!")
        glove_matrix = torch.randn(len(token2idx), config.glove_dim) * 0.01
        glove_matrix[0] = 0  # PAD = zero

    # 2. Build model + load checkpoint
    print("\n--- Loading model ---")
    model = SeqTRDet(config, glove_matrix).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Ưu tiên dùng EMA weights nếu có
    if 'ema_shadow' in ckpt:
        print("Using EMA weights")
        model.load_state_dict(ckpt['ema_shadow'], strict=True)
    else:
        model.load_state_dict(ckpt['model_state_dict'], strict=True)

    epoch = ckpt.get('epoch', '?')
    print(f"Loaded checkpoint from epoch {epoch}")

    # 3. Evaluate trên từng split
    print("\n" + "=" * 60)
    results = {}
    # Mới
    for split in args.splits:
        print(f"\n--- Evaluating on [{split}] ---")
        # Xác định đường dẫn file JSON tùy theo split
        if split == 'train':
            ann_file = config.train_ann_file
        elif split == 'val':
            ann_file = config.val_ann_file
        else:
            ann_file = getattr(config, f"{split}_ann_file", None)
            if ann_file is None:
                print(f"  ⚠️ Cấu hình không có file annotation cho split '{split}'. Bỏ qua.")
                continue
        try:
            dataset = CustomGroundingDataset(
                ann_file, config.img_dir, split,
                token2idx, config.max_token, config.img_size
            )
            loader = build_dataloader(
                dataset, batch_size=args.batch_size,
                shuffle=False, num_workers=config.num_workers
            )
            acc, avg_iou = evaluate(model, loader, device, desc=split)
            results[split] = {'accuracy': acc, 'avg_iou': avg_iou}
        except KeyError:
            print(f"  ⚠️ Split '{split}' không tồn tại trong file annotations.")

    # 4. Tổng kết
    print("\n" + "=" * 60)
    print("TỔNG KẾT KẾT QUẢ")
    print("=" * 60)
    for split, res in results.items():
        print(f"  {split:8s}: Acc@IoU>=0.5 = {res['accuracy']:.2f}% | "
              f"Avg IoU = {res['avg_iou']:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
