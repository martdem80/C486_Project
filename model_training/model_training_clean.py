import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
from PIL import Image
import torchvision.transforms.functional as TF


class SignLanguageDataset(Dataset):
    """
    Handles data loading, preprocessing, and augmentation of sign language images.
    """
    def __init__(self, data, labels, transform=None, augment=False):
        self.data = data
        self.label_map = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        self.labels = np.array([self.label_map[label] for label in labels])
        self.transform = transform
        self.augment = augment
        self.num_classes = len(self.label_map)

        self.rotation_range = (-15, 15)
        self.scale_range = (0.9, 1.1)
        self.brightness_range = (0.8, 1.2)
        self.contrast_range = (0.8, 1.2)

    def __len__(self):
        return len(self.labels)

    def custom_transform(self, image):
        if not self.augment:
            return image

        if isinstance(image, torch.Tensor):
            image = TF.to_pil_image(image)

        if random.random() > 0.5:
            angle = random.uniform(*self.rotation_range)
            image = TF.rotate(image, angle)

        if random.random() > 0.5:
            scale = random.uniform(*self.scale_range)
            new_size = tuple(int(s * scale) for s in image.size)
            image = TF.resize(image, new_size)
            image = TF.resize(image, image.size)

        if random.random() > 0.5:
            brightness_factor = random.uniform(*self.brightness_range)
            image = TF.adjust_brightness(image, brightness_factor)

        if random.random() > 0.5:
            contrast_factor = random.uniform(*self.contrast_range)
            image = TF.adjust_contrast(image, contrast_factor)

        image = TF.to_tensor(image)
        return image

    def __getitem__(self, idx):
        # Reshape flat array into 28x28 image and normalize
        image = self.data[idx].reshape(28, 28).astype(np.float32)
        image = image / 255.0

        # Convert to 3 channels for MobileNetV3
        image = np.stack([image] * 3, axis=0)
        image = torch.FloatTensor(image)

        image = self.custom_transform(image)

        if self.transform:
            image = self.transform(image)

        label = int(self.labels[idx])
        return image, label

    def get_num_classes(self):
        return self.num_classes


def create_transforms(is_training=True):
    """Create transformation pipelines for training and validation data."""
    if is_training:
        transform = transforms.Compose([
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    shear=5
                )
            ], p=0.5),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.2))
            ], p=0.3),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    return transform


def load_and_prepare_data(data_path):
    """Load and split dataset into training and validation sets."""
    data = pd.read_csv(data_path)
    unique_labels = sorted(data['label'].unique())
    print(f"Unique labels: {unique_labels}")
    print(f"Number of classes: {len(unique_labels)}")

    X = data.drop('label', axis=1).values
    y = data['label'].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_val, y_train, y_val


def create_model(num_classes):
    """Configure MobileNetV3-Small model for the task."""
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

    model.classifier = nn.Sequential(
        nn.Linear(576, 1024),
        nn.Hardswish(),
        nn.Dropout(p=0.3),
        nn.Linear(1024, num_classes)
    )
    return model


def train_model(model, train_loader, val_loader, device, num_epochs=30):
    """Train the model using provided data loaders."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=3,
        factor=0.5,
        min_lr=1e-6
    )

    best_train_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * train_correct / train_total

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')

        scheduler.step(val_acc)

        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'best_train_acc': best_train_acc
            }, 'best_train_model.pth')

    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    X_train, X_val, y_train, y_val = load_and_prepare_data('sign_mnist_train.csv')

    train_dataset = SignLanguageDataset(
        X_train, y_train,
        transform=create_transforms(is_training=True),
        augment=True
    )
    val_dataset = SignLanguageDataset(
        X_val, y_val,
        transform=create_transforms(is_training=False),
        augment=False
    )

    num_classes = train_dataset.get_num_classes()
    print(f"Number of classes in dataset: {num_classes}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = create_model(num_classes)
    model = model.to(device)

    train_model(model, train_loader, val_loader, device)
    print("Training completed!")


if __name__ == "__main__":
    main()