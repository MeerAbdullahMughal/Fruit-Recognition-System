from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_path = r"C:\Users\Hp\OneDrive\Desktop\Python codes\intern tasks\1st DL project (CNN)\train"
val_path = r"C:\Users\Hp\OneDrive\Desktop\Python codes\intern tasks\1st DL project (CNN)\validation"
test_path = r"C:\Users\Hp\OneDrive\Desktop\Python codes\intern tasks\1st DL project (CNN)\test"

train_dataset = ImageFolder(train_path, transform=transform)
val_dataset = ImageFolder(val_path, transform=transform)
test_dataset = ImageFolder(test_path, transform=transform)

print("Classes:", train_dataset.classes)
print("Class Mapping:", train_dataset.class_to_idx)
print("Training Images:", len(train_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32
)

for images, labels in train_loader:
    print(images.shape)
    print(labels)
    break