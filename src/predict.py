import torch
from torchvision import transforms
from PIL import Image

from model import get_model

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes (IMPORTANT: same order as training)
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Load model
model = get_model(num_classes=4)
model.load_state_dict(torch.load("../models/resnet50_brain_tumor.pth"))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)

    return class_names[pred.item()]


if __name__ == "__main__":
    img_path = input("Enter image path: ")
    result = predict_image(img_path)
    print("Prediction:", result)