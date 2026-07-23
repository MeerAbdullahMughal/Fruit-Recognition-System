from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

# =====================================================
# IMAGE TRANSFORMS
# =====================================================

transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =====================================================
# DATASET PATHS
# =====================================================

train_path = r"C:\Users\Hp\OneDrive\Desktop\Python codes\intern tasks\1st DL project (CNN)\train"
val_path = r"C:\Users\Hp\OneDrive\Desktop\Python codes\intern tasks\1st DL project (CNN)\validation"
test_path = r"C:\Users\Hp\OneDrive\Desktop\Python codes\intern tasks\1st DL project (CNN)\test"

# =====================================================
# LOAD DATASETS
# =====================================================

train_dataset = ImageFolder(train_path, transform=transform)
val_dataset = ImageFolder(val_path, transform=transform)
test_dataset = ImageFolder(test_path, transform=transform)

print("Classes:", train_dataset.classes)
print("Class Mapping:", train_dataset.class_to_idx)

print("Training Images:", len(train_dataset))
print("Validation Images:", len(val_dataset))
print("Testing Images:", len(test_dataset))

# =====================================================
# DATALOADERS
# =====================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# Show one batch

images, labels = next(iter(train_loader))

print("Image Batch Shape:", images.shape)
print("Labels:", labels)

# =====================================================
# DEVICE
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# =====================================================
# CNN MODEL
# =====================================================

class FruitCNN(nn.Module):

    def __init__(self, num_classes):
        super(FruitCNN, self).__init__()

        self.network = nn.Sequential(

            # --------------------------
            # Block 1
            # --------------------------

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # --------------------------
            # Block 2
            # --------------------------

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # --------------------------
            # Block 3
            # --------------------------

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # --------------------------
            # Block 4
            # --------------------------

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Reduce feature maps to 1x1
            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),

            # Fully Connected Layers

            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, num_classes)

        )

    def forward(self, x):
        return self.network(x)

# =====================================================
# CREATE MODEL
# =====================================================

num_classes = len(train_dataset.classes)

model = FruitCNN(num_classes).to(device)

print(model)

# =====================================================
# LOSS FUNCTION
# =====================================================

criterion = nn.CrossEntropyLoss()

# =====================================================
# OPTIMIZER
# =====================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# =====================================================
# TRAINING
# =====================================================

num_epochs = 10

for epoch in range(num_epochs):

    print(f"\n========== STARTING EPOCH {epoch+1} ==========")

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        print(f"Processing Batch {batch_idx + 1}/{len(train_loader)}")

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    average_loss = running_loss / len(train_loader)

    # =====================================================
    # VALIDATION
    # =====================================================

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total

    print(
        f"Epoch [{epoch+1}/{num_epochs}] | "
        f"Loss: {average_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}% | "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )

# =====================================================
# TESTING
# =====================================================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total

print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")

# =====================================================
# SAVE MODEL
# =====================================================

torch.save(
    model.state_dict(),
    r"C:\Users\Hp\OneDrive\Desktop\Python codes\intern tasks\fruit_cnn.pth"
)

print("Model saved successfully!")
