import torch 
import torch.nn as nn
import torch.nn.functional as F

class SimpleFusion(nn.Module):
    """
    Kết hợp vis_feature(C3,C4,C5) + lang_feature
    Input :
        vis_feature: list[tensor] có 3 tensor (c3,c4,c5) với shape (B,C,H,W) khác nhau C3[B,512,80,80] C4[B,1024,40,40] C5[B,2048,20,20]
        lang_feature: tensor có shape (B,1,1024) được lấy từ last layerGRU

    Output :
        Tensor có shape (B,1024,H,W) đã được nhân hàm tanh với lang_feature
    """
    def __init__(self, vis_channels = [512,1024,2048]):
        super().__init__()
        c3_ch, c4_ch, c5_ch = vis_channels

        # chuyển c3 từ 80*80 -> 40 * 40, giữ nguyên channel
        self.down_c3 = nn.Sequential(
            nn.Conv2d(in_channels=c3_ch, out_channels=c3_ch, kernel_size=3,stride=2,padding =1, bias = False),
            nn.BatchNorm2d(c3_ch),
            nn.ReLU(inplace=True)
        )
        
        # concat c3 và c4 lại sau đó chuyển xuống c4 từ 40*40 -> 20*20
        mid_ch = c3_ch + c4_ch # 512 + 1024 = 1536
        self.down_mid = nn.Sequential(
            nn.Conv2d(in_channels=mid_ch, out_channels=mid_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace = True)
        )

        # sau bước down_c4 thì feature sẽ có shape [B, 1536, 20,20]
        # ta sẽ con cat với c5 thì sẽ có shape [B, 1536+2048, 20, 20]
        # chúng ta sẽ project về thành [B,1024,20,20]
        merged_ch = mid_ch + c5_ch # 1536 + 2048 = 3584
        out_ch = 1024
        self.project = nn.Sequential(
            nn.Conv2d(in_channels=merged_ch, out_channels= out_ch, kernel_size =3, stride =1, padding=1, bias = False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=1, bias= False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, vis_feats, lang_feat):
        c3,c4,c5 = vis_feats
        c3_down = self.down_c3(c3)
        mid = torch.cat([c3_down,c4], dim =1)
        mid_down = self.down_mid(mid)
        merged = torch.cat([mid_down,c5],dim=1)
        x_vis = self.project(merged) # ta đã có visual feature, shape [B,1024,20,20]

        # lange_feat có shape [B,1,1024], muốn nhân được thì ta phải chuyển nó thành shape [B,1024,1,1]
        y = lang_feat.squeeze(1).unsqueeze(-1).unsqueeze(-1)

        x_fused = torch.tanh(x_vis)*torch.tanh(y)
        
        return x_fused

if __name__ == "__main__":
    print("=== Test SimpleFusion ===")

    model = SimpleFusion(vis_channels=[512, 1024, 2048])

    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,}")

    B = 2
    c3 = torch.randn(B, 512, 80, 80)
    c4 = torch.randn(B, 1024, 40, 40)
    c5 = torch.randn(B, 2048, 20, 20)
    lang = torch.randn(B, 1, 1024)

    x_fused = model([c3, c4, c5], lang)
    print(f"Input:  C3={c3.shape}, C4={c4.shape}, C5={c5.shape}, lang={lang.shape}")
    print(f"Output: {x_fused.shape}")  # Expected: [2, 1024, 20, 20]
    print(f"Range:  [{x_fused.min():.3f}, {x_fused.max():.3f}]")

    assert x_fused.max() <= 1.0 and x_fused.min() >= -1.0, "Range ngoài [-1,1]!"
    print("✅ SimpleFusion test passed!")
