import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.model import get_model

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

st.markdown("""
<style>
/* Background */
.main {
    background-color: #ffcc80;
}

/* Title */
.title {
    text-align: center;
    color: #331f00;
    font-size: 40px;
    font-weight: bold;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #995c00;
    margin-bottom: 20px;
}

/* Card */
.card {
    background-color: #fff5e6;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 10px #ffebcc;
    margin-bottom: 20px;
}

/* Prediction box */
.prediction {
    font-size: 22px;
    color: #331f00;
    font-weight: bold;
}

.confidence {
    font-size: 18px;
    color: #663d00;
}

/* Center everything */
.center {
    text-align: center;
}
            
.styled-text {
    color: #804d00;    
}
</style>
""", unsafe_allow_html=True)


# Title
st.markdown('<div class="title">Brain Tumor Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered MRI classification with explainability</div>', unsafe_allow_html=True)


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

def generate_gradcam(model, image, input_tensor):
    gradients = None
    activations = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    target_layer = model.layer4[-1]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    # Forward
    output = model(input_tensor)
    pred_class = output.argmax(dim=1)

    # Backward
    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients.cpu().data.numpy()[0]
    acts = activations.cpu().data.numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    # Overlay
    img = np.array(image.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = heatmap * 0.4 + img

    forward_handle.remove()
    backward_handle.remove()

    return overlay.astype(np.uint8)

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

    

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)

    probs = torch.nn.functional.softmax(outputs, dim=1)
    confidence, pred = torch.max(probs, 1)

    result = class_names[pred.item()]
    confidence = confidence.item() * 100

    st.markdown(f'''
    <div class="card center">
        <div class="prediction">Prediction: {result}</div>
        <div class="confidence">Confidence: {confidence:.2f}%</div>
    </div>
    ''', unsafe_allow_html=True)

    # 🔥 Grad-CAM
    cam_image = generate_gradcam(model, image, input_tensor)

    

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, width=300)
        st.markdown("<p class='styled-text' style='text-align:center;'>Original MRI</p>", unsafe_allow_html=True)

    with col2:
        st.image(cam_image, width=300)
        st.markdown("<p class='styled-text' style='text-align:center;'>Grad-CAM</p>", unsafe_allow_html=True)

   

st.markdown("<hr><center>Developed with ❤️ using Deep Learning</center>", unsafe_allow_html=True)