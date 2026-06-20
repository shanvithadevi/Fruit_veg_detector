# app.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Define model (same as training)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 30 * 30)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# Load model
model = CNNModel()
model.load_state_dict(torch.load("model.h5", map_location=torch.device("cpu")))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

st.title("🍎 Fruits & Vegetables Quality Check")

uploaded_file = st.file_uploader("Upload an image (PNG or JPG)", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        prediction = model(img_tensor).item()
    
    confidence = prediction if prediction > 0.5 else 1 - prediction
    if prediction > 0.5:
        st.error(f"❌ Spoiled Produce Detected (Confidence: {confidence*100:.2f}%)")
    else:
        st.success(f"✅ Fresh Produce Detected (Confidence: {confidence*100:.2f}%)")
