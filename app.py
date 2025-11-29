import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np


# -----------------------------------------------------------
# 1. 定義模型結構 (必須與訓練時完全一致)
# -----------------------------------------------------------
class ConvBN(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=3, padding='same')
        self.bn = nn.BatchNorm2d(num_features=cout)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class CNN(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBN(3, 16),
            ConvBN(16, 16),
            nn.MaxPool2d(kernel_size=2),

            ConvBN(16, 32),
            ConvBN(32, 32),
            nn.MaxPool2d(kernel_size=2),

            ConvBN(32, 64),
            ConvBN(64, 64),
            nn.MaxPool2d(kernel_size=2),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64 * 32 * 32, 256),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


# -----------------------------------------------------------
# 2. 設定與加載模型 (使用 cache 避免重複加載)
# -----------------------------------------------------------
# 設定標籤字典
LABEL_DICT = {
    0: 'Normal',
    1: 'Bacteria',
    2: 'Virus'
}


@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN(kernel_size=7)

    # 確保路徑正確，如果是上傳到雲端，best2.pth 需要在同一目錄下
    # map_location='cpu' 是為了防止 Server 沒有 GPU 時報錯
    model.load_state_dict(torch.load('best2.pth', map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    return model, device


# -----------------------------------------------------------
# 3. Streamlit 介面邏輯
# -----------------------------------------------------------
st.title("肺部 X-ray 分類器 (Pneumonia Detection)")
st.write("請上傳一張 X-ray 圖片進行檢測")

# 加載模型
try:
    model, device = load_model()
except FileNotFoundError:
    st.error("錯誤：找不到模型檔案 'best2.pth'。請確認檔案已上傳。")
    st.stop()

# 檔案上傳器
uploaded_file = st.file_uploader("選擇圖片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- A. 讀取圖片 ---
    # 將上傳的檔案轉換為 OpenCV 格式
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)  # BGR 格式，與 cv2.imread 相同

    # 顯示圖片 (Streamlit 需要 RGB，所以要轉一下給人類看)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='上傳的圖片', width='stretch')

    # --- B. 預處理 (與你的原始代碼一致) ---
    image_resized = cv2.resize(image, (256, 256))

    # 轉換為 tensor 並正規化
    tensor = torch.from_numpy(image_resized.astype(np.float32) / 255.0)

    # 維度調整 (H, W, C) -> (C, H, W)
    tensor = tensor.permute(2, 0, 1)

    # 增加 Batch 維度 -> (1, C, H, W)
    tensor = tensor.unsqueeze(0)

    # 移動到設備
    tensor = tensor.to(device)

    # --- C. 預測 ---
    if st.button('開始分析'):
        with torch.no_grad():
            output = model(tensor)
            prediction = torch.argmax(output, dim=1).item()
            probability = torch.softmax(output, dim=1)[0, prediction].item()

        # --- D. 顯示結果 ---
        label_name = LABEL_DICT[prediction]

        # 根據結果顯示不同顏色
        if prediction == 0:
            st.success(f"預測結果: {label_name} (正常)")
        else:
            st.error(f"預測結果: {label_name} (異常)")

        st.info(f"置信度 (Confidence): {probability:.4f}")