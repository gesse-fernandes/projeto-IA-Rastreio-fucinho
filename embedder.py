# embedder.py
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np

class ResnetEmbedder(nn.Module):
    def __init__(self, weights="IMAGENET"):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if weights=="IMAGENET" else None)
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # até global avgpool
        self.out_dim = 512
        self.transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        self.eval()

    @torch.inference_mode()
    def embed_pil(self, pil_img: Image.Image) -> np.ndarray:
        x = self.transform(pil_img).unsqueeze(0)
        feat = self.backbone(x).flatten(1)  # [1,512]
        v = feat[0].cpu().numpy()
        # L2-normalize p/ FAISS
        v = v / (np.linalg.norm(v) + 1e-9)
        return v.astype("float32")
