class Config:
    img_dir = "/kaggle/input/datasets/minhkhoai/personal-grounding/data/images/images"
    train_ann_file = "/kaggle/input/datasets/minhkhoai/personal-grounding/data/train.json"
    dev_ann_file = "/kaggle/input/datasets/minhkhoai/personal-grounding/data/dev.json"
    test_ann_file = "/kaggle/input/datasets/minhkhoai/personal-grounding/data/test.json"

    # Đường dẫn tới file Word2Vec tiếng Việt đã tải
    w2v_path = "/kaggle/input/datasets/minhkhoai/personal-grounding/data/word2vec_vi_words_300dims/word2vec_vi_words_300dims.txt"
    w2v_is_binary = False  # Đặt True nếu là tệp .bin, False nếu là tệp .vec/.txt
    glove_dim = 300       # Số chiều của vector Word2Vec (thường là 300)

    img_size = 640

    max_token = 15


    # Tham số cho visual backbone
    backbone_out_channels = 1024 # số channel đầu ra của backbone (ResNet-50)

    # Tham số cho language branch
    gru_hidden = 512 

    # Tham số cho kiến trúc Transformer
    d_model = 256
    nhead = 8

    dim_feedforward = 1024 

    enc_layers = 6
    dec_layers = 3

    dropout = 0.1

    # Quantization
    num_bin = 1000
    vocab_size = 1001 # thêm 1 token cho End token

    # Tham số cho training
    batch_size = 16
    lr = 6.25e-5
    epochs = 60
    warmup_epochs = 5 # Trong 5 epoch đầu, lr tăng từ 0 lên lr giúp model ổn định giai đoạn đầu
    decay_epoch = 50 # Epoch mà lr sẽ giảm (nhân với ratio)
    decay_ratio = 0.1
    grad_clip = 0.15 # Giói hạn norm gradient để tránh exploding gradients, nếu >0.15 thì giảm thành 0.15

    # EMA duy trì 1 bản sao "trung bình" của model weights.
    # ema_decay = 0.999 nghĩa là: shadow = 0.999 * shadow + 0.001 * current_weights
    ema = True
    ema_decay = 0.999

    # Thay vì hard target [0, 0, 1, 0, ...], dùng soft target [0.0001, 0.0001, 0.9, 0.0001, ...]
    # Giúp model tránh over-confident và generalize tốt hơn.
    label_smoothing = 0.1

    # 6. LOGGING & CHECKPOINT
    # In log mỗi N batches.
    log_interval = 80
    # Random seed: Đảm bảo kết quả reproducible (chạy lại ra cùng kết quả).
    seed = 6666
    # Số workers cho DataLoader. Kaggle nên dùng 0 để tránh memory leak qua nhiều epoch.
    # [CŨ] num_workers = 2
    num_workers = 2
    # Thư mục lưu checkpoint & log.
    work_dir = "/kaggle/working/checkpoints"


    


