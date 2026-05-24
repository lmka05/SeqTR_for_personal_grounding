import torch 
import torch.nn as nn

class LanguageEncoder(nn.Module):
    def __init__(self, glove_vectors, hidden_size = 512):
        super().__init__()
        self.hidden_size = hidden_size
        vocab_size, embed_dim = glove_vectors.shape

        self.embedding = nn.Embedding.from_pretrained(
            glove_vectors,
            padding_idx=0,
            freeze = True # Không train glove

        )
        
        self.gru = nn.GRU(
            input_size = embed_dim,
            hidden_size = hidden_size,
            num_layers =1,
            bidirectional = True,
            batch_first = True,
            bias = True,
            dropout = 0.0, # chỉ có 1 layer trên không dropout
        )

    def forward(self, ref_inds):
        # ref_inds: [B, max_lang_tokens]
        mask = (ref_inds == 0) # mask trả true cho token 0 (đây là mặt nạ để nó biết chỗ nào là mask)

        emb = self.embedding(ref_inds) # tra bảng embedding để lấy ra vector cho mỗi token

        output,_ = self.gru(emb) # ouput : output tại mỗi token (512 chiều), do nó có 2 hướng nên shape của output là [B, seq_len, hidden_size * 2]
        
        # hàm mask_fill(condition,value) : gán value tại những chỗ mà condition true, hàm unsqueeze(position) này sẽ thêm chiều 1 vào vị trí được truyền vào
        # trước khi max pooling thì mình cần phải gán âm vô cùng cho vị trí pad thì nó sẽ không bao giờ thắng
        output = output.masked_fill(mask.unsqueeze(-1), float('-inf'))

        y = output.max(dim =1, keepdim = True).values # với mỗi chiều trong 1024 chiều, thì nó sẽ lấy max trong 15 token
        
        return y
