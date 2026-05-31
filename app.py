import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
import gdown
import os

st.set_page_config(
    page_title="Brain Tumor Detection AI",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor Detection AI")
st.markdown("**MRI aur CT scan se tumor detect karta hai**")
st.markdown("---")

@st.cache_resource
def load_models():
    if not os.path.exists("best_classifier.pth"):
        with st.spinner("Classification model load ho raha hai..."):
            gdown.download(
                "https://drive.google.com/uc?id=1KhqSHfWTerUSWibA1deAasVkyhkRr-f6",
                "best_classifier.pth", quiet=False
            )
    if not os.path.exists("best_segmentation.pth"):
        with st.spinner("Segmentation model load ho raha hai..."):
            gdown.download(
                "https://drive.google.com/uc?id=1ftwFn5Jzhy8qDqrXJut92OsVkerVKL5E",
                "best_segmentation.pth", quiet=False
            )

    device = torch.device('cpu')

    def double_conv(a, b):
        return nn.Sequential(
            nn.Conv2d(a,b,3,padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
            nn.Conv2d(b,b,3,padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1=double_conv(1,64);    self.e2=double_conv(64,128)
            self.e3=double_conv(128,256); self.e4=double_conv(256,512)
            self.pool=nn.MaxPool2d(2)
            self.bot=double_conv(512,1024)
            self.u4=nn.ConvTranspose2d(1024,512,2,2); self.d4=double_conv(1024,512)
            self.u3=nn.ConvTranspose2d(512,256,2,2);  self.d3=double_conv(512,256)
            self.u2=nn.ConvTranspose2d(256,128,2,2);  self.d2=double_conv(256,128)
            self.u1=nn.ConvTranspose2d(128,64,2,2);   self.d1=double_conv(128,64)
            self.out=nn.Conv2d(64,1,1)
        def forward(self,x):
            e1=self.e1(x); e2=self.e2(self.pool(e1))
            e3=self.e3(self.pool(e2)); e4=self.e4(self.pool(e3))
            b=self.bot(self.pool(e4))
            x=self.d4(torch.cat([self.u4(b),e4],1))
            x=self.d3(torch.cat([self.u3(x),e3],1))
            x=self.d2(torch.cat([self.u2(x),e2],1))
            x=self.d1(torch.cat([self.u1(x),e1],1))
            return torch.sigmoid(self.out(x))

    cls_model = models.efficientnet_b3(weights=None)
    cls_model.classifier[1] = nn.Linear(cls_model.classifier[1].in_features, 4)
    cls_model.load_state_dict(torch.load("best_classifier.pth", map_location=device))
    cls_model.eval()

    seg_model = UNet()
    seg_model.load_state_dict(torch.load("best_segmentation.pth", map_location=device))
    seg_model.eval()

    return cls_model, seg_model, device

cls_model, seg_model, device = load_models()

class_names = {
    0: '✅ CT Scan — Healthy',
    1: '🔴 CT Scan — Tumor Detected',
    2: '✅ MRI Scan — Healthy',
    3: '🔴 MRI Scan — Tumor Detected'
}

st.subheader("📤 Scan Upload karo")
uploaded_file = st.file_uploader(
    "MRI ya CT scan image select karo",
    type=['jpg','jpeg','png']
)

if uploaded_file:
    img_pil = Image.open(uploaded_file).convert('RGB')
    st.image(img_pil, caption="Uploaded Scan", use_column_width=True)

    with st.spinner("🤖 AI analyze kar raha hai..."):
        val_tfm = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        inp = val_tfm(img_pil).unsqueeze(0)
        with torch.no_grad():
            out  = cls_model(inp)
            idx  = out.argmax(1).item()
            conf = torch.softmax(out,1).max().item()
            name = class_names[idx]

        img_gray = img_pil.convert('L').resize((256,256))
        gray     = np.array(img_gray) / 255.0
        seg_inp  = torch.tensor(gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            mask = seg_model(seg_inp).squeeze().numpy()

    st.markdown("---")
    st.subheader("📊 Result")

    if 'Tumor' in name:
        st.error(f"### {name}")
    else:
        st.success(f"### {name}")

    st.metric("Confidence", f"{conf:.1%}")

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_pil, caption="Original Scan", use_column_width=True)
    with col2:
        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(gray, cmap='gray')
        ax.imshow(mask > 0.5, cmap='Reds', alpha=0.5)
        ax.axis('off')
        ax.set_title('Tumor Location')
        st.pyplot(fig)

    st.markdown("---")
    st.warning("⚠️ Yeh AI research tool hai — final diagnosis ke liye doctor se zaroor milein!")
