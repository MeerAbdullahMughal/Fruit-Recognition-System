import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

# ======================================
# PATHS
# ======================================

train_path = r"C:\Users\Hp\OneDrive\Desktop\Python codes\intern tasks\1st DL project (CNN)\train"

model_path = r"C:\Users\Hp\OneDrive\Desktop\Python codes\intern tasks\fruit_cnn.pth"

# ======================================
# DEVICE
# ======================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using:", device)

# ======================================
# LOAD CLASS NAMES
# ======================================

train_dataset = ImageFolder(train_path)

classes = train_dataset.classes

print("Classes:", classes)

# ======================================
# CNN MODEL
# ======================================

class FruitCNN(nn.Module):

    def __init__(self, num_classes):
        super(FruitCNN, self).__init__()

        self.network = nn.Sequential(

            # Convolution Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Convolution Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Convolution Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Same architecture as training
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# ======================================
# LOAD MODEL
# ======================================

model = FruitCNN(len(classes)).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))

model.eval()

print("Model loaded successfully!")

# ======================================
# IMAGE TRANSFORM
# ======================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ======================================
# GET IMAGE PATH
# ======================================

# ======================================
# GET IMAGE PATH OR URL
# ======================================

from urllib.request import urlopen

image_path = input("\nEnter image path or URL: ")

try:
    if image_path.startswith("http://") or image_path.startswith("https://"):
        image = Image.open(urlopen(image_path)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")

except Exception as e:
    print("Error:", e)
    exit()

# ======================================
# PREPROCESS IMAGE
# ======================================

image = transform(image)
image = image.unsqueeze(0)
image = image.to(device)

# ======================================
# PREDICT
# ======================================

with torch.no_grad():

    output = model(image)

    probabilities = torch.softmax(output, dim=1)

    confidence, predicted = torch.max(probabilities, 1)

# ======================================
# RESULT
# ======================================

print("\n========== RESULT ==========")
print("Predicted Fruit :", classes[predicted.item()])
print(f"Confidence      : {confidence.item() * 100:.2f}%")
print("============================")