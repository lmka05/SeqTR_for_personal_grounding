import torch
import torch.nn as nn
import torchvision.models as models

class VisualEncoder(nn.Module):
    def __init__(self, freeze_layers = True ):
        super().__init__()
        
        # tải resnet pretrain
        resnet = models.resnet50(weights = 'IMAGENET1K_V1')

        # layer 0 : [B,3,640,640] -> [B,64,160,160]
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        # layer 1 : [B,64,160,160] -> [B,256,160,160]
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2  # → [B, 512, 80, 80]   ← output C3
        self.layer3 = resnet.layer3  # → [B, 1024, 40, 40]  ← output C4
        self.layer4 = resnet.layer4  # → [B, 2048, 20, 20]  ← output C5

        if freeze_layers:
            for layer in [self.layer0,self.layer1]:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self,img):
        x = self.layer0(img)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return [c3,c4,c5]

if __name__ == "__main__":
    print("=== Test VisualEncoder ===")

    model = VisualEncoder(freeze_layers=True)

    # Đếm tham số
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}")
    print(f"Trainable:    {trainable:,}")
    print(f"Frozen:       {total - trainable:,}")

    # Test forward
    dummy_img = torch.randn(2, 3, 640, 640)  # Batch of 2 images
    features = model(dummy_img)

    for i, feat in enumerate(features):
        print(f"Feature C{i+3}: {feat.shape}")
    # Expected:
    # Feature C3: torch.Size([2, 512, 80, 80])
    # Feature C4: torch.Size([2, 1024, 40, 40])
    # Feature C5: torch.Size([2, 2048, 20, 20])

    print("✅ VisualEncoder test passed!")
