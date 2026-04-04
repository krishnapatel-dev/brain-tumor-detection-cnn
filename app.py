import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

from src.model import get_model

# Page title
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image to classify tumor type")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Load model
@st.cache_resource
def load_model():
    model = get_model(num_classes=4)
    model.load_state_dict(torch.load("models/resnet50_brain_tumor.pth"))
    model.eval()
    return model

model = load_model()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Upload image
uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)

    result = class_names[pred.item()]

    st.success(f"Prediction: {result}")