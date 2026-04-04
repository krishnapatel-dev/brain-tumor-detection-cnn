import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import get_data_loaders
from model import get_model

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
EPOCHS = 15
LEARNING_RATE = 0.0001

def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    return correct

def train():

    # Load data
    train_loader, val_loader, class_names = get_data_loaders("../data")

    # Load model
    model = get_model(num_classes=len(class_names))

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer (only train last layer)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0001
    )

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            correct_train += calculate_accuracy(outputs, labels)
            total_train += labels.size(0)

        train_acc = 100 * correct_train / total_train

        # 🔥 Validation
        model.eval()
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                correct_val += calculate_accuracy(outputs, labels)
                total_val += labels.size(0)

        val_acc = 100 * correct_val / total_val

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss:.2f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    print("Training Complete!")
    torch.save(model.state_dict(), "../models/resnet50_brain_tumor.pth")
    print("Model saved!")

if __name__ == "__main__":
    train()