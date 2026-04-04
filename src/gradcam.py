import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


from model import get_model

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Load model
model = get_model(num_classes=4)
model.load_state_dict(torch.load("../models/resnet50_brain_tumor.pth"))
model.eval()

# Target layer (VERY IMPORTANT)
target_layer = model.layer4[-1]

# Hook for gradients
gradients = None

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

# Hook for activations
activations = None

def forward_hook(module, input, output):
    global activations
    activations = output

target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def generate_gradcam(image_path):

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Forward
    output = model(input_tensor)
    pred_class = output.argmax(dim=1)

    # Backward
    model.zero_grad()
    output[0, pred_class].backward()

    # Convert to numpy
    grads = gradients.cpu().data.numpy()[0]
    acts = activations.cpu().data.numpy()[0]

    # Compute weights
    weights = np.mean(grads, axis=(1, 2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    # Overlay
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = heatmap * 0.4 + img

    # Show
    plt.imshow(cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.title(f"Prediction: {class_names[pred_class.item()]}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    path = input("Enter image path: ").strip().strip('"')
    generate_gradcam(path)