import re
import json
import numpy as np
import torch
from pyvi import ViTokenizer
import os               

def clean_expression(expression):
    ''''
    Làm sạch câu : xoá các ký tự đặc biệt, lower, tách thành list các từ
    '''
    expression_cleaned = re.sub(r"([.,'!?\"()*#:;])", '', expression.lower())

    expression_cleaned = expression_cleaned.replace('-', ' ')

    expression_cleaned = expression_cleaned.replace('/',' ')

    expression_tokenized = ViTokenizer.tokenize(expression_cleaned)
    return expression_tokenized.split()

def build_vocab(ann_files):
    """
    Xây dựng vocabulary (bảng từ vựng) từ danh sách các file annotations tùy chỉnh.
    Args:
        ann_files (str | list): Đường dẫn đơn hoặc danh sách đường dẫn đến file JSON.
    """
    token2idx = {
        "PAD": 0,
        "UNK": 1,
    }
    
    # Nếu truyền vào 1 đường dẫn dạng chuỗi đơn lẻ, chuyển thành list để xử lý chung
    if isinstance(ann_files, str):
        ann_files = [ann_files]

    for ann_file in ann_files:
        if not os.path.exists(ann_file):
            print(f"⚠️ Warning: File {ann_file} không tồn tại để build vocab.")
            continue
            
        with open(ann_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Duyệt qua cấu trúc cây JSON để trích xuất từ vựng
        for img_id, img_info in data.items():
            if not isinstance(img_info, 'dict'):
                continue

            bboxes = img_info.get('bboxes', [])
            if bboxes is None: continue
            
            for bbox_info in bboxes:
                if not isinstance(bbox_info, dict): continue
                
                # Ép kiểu description về List giống như bên Dataset
                expressions = bbox_info.get('description', [])
                if isinstance(expressions, str):
                    expressions = [expressions]
                    
                for expression in expressions:
                    # Bỏ qua nếu là chuỗi rỗng
                    if not isinstance(expression, str) or not expression.strip():
                        continue
                        
                    words = clean_expression(expression)
                    for word in words:
                        if word not in token2idx:
                            token2idx[word] = len(token2idx)
    idx2token = {idx: token for token, idx in token2idx.items()}
    return token2idx, idx2token

def build_w2v_matrix(token2idx, w2v_model, w2v_dim = 300):
    """
    Tạo ma trận embedding Word2Vec cho vocabulary.

    Mỗi từ trong vocab sẽ được map sang vector Word2Vec 300 chiều.
    Từ nào không có trong Word2Vec → dùng vector ngẫu nhiên.

    Args:
        token2idx (dict): Vocabulary đã build. Ví dụ: {"PAD": 0, "the": 3, ...}
        w2v_model: Model GloVe đã tải (từ gensim).
        w2v_dim (int): Chiều của GloVe vector (300).

    Returns:
        weight_matrix (Tensor): [vocab_size, 300], dùng để khởi tạo nn.Embedding.

    Ví dụ:
        Vocab có 5000 từ → output shape = [5000, 300]
        weight_matrix[0] = vector của "PAD" = toàn số 0
        weight_matrix[3] = vector GloVe của "the" = [0.04, -0.2, ...]
    """
    vocab_size = len(token2idx)

    # Khởi tạo ma trận cho các từ không có trong glove (số nhỏ)
    weight_matrix = np.random.uniform(-0.01,0.01, (vocab_size, w2v_dim)).astype(np.float32)
    weight_matrix[0] = np.zeros(w2v_dim) # PAD token

    # Đếm sô từ kiếm được trong glove
    found =0
    for word, idx in token2idx.items():
        if word in w2v_model:
            weight_matrix[idx] = w2v_model[word]
            found += 1
    print(f"Tìm được {found} trên {vocab_size} từ trong w2v")

    return torch.from_numpy(weight_matrix)

def tokenize_expression(expression, token2idx, max_token) :
    """
    Chuyển 1 câu referring expression thành tensor các index.
    """
    # Tạo tensor toàn 0 (PAD) với kích thước max_token
    ref_inds = torch.zeros(max_token, dtype=torch.long)

    # Tách câu thành các từ đã làm sạch
    words = clean_expression(expression)

    for i, word in enumerate(words):
        # Dừng nếu đã đủ max_token từ (truncation)
        if i >= max_token:
            break

        if word in token2idx:
            # Từ có trong vocab → dùng index của nó
            ref_inds[i] = token2idx[word]
        else:
            # Từ lạ → dùng UNK token (index 1)
            ref_inds[i] = token2idx['UNK']

    return ref_inds