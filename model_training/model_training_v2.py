import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import numpy as np
from tqdm import tqdm
import random
import os
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import torchvision.transforms.functional as TF


class AdvancedAugmentation:
    """
    Advanced data augmentation techniques for sign language recognition.
    """

    def __init__(self, prob=0.5, num_classes=25):  # Updated to 25 classes (24 letters + '_')
        self.prob = prob
        self.num_classes = num_classes

    def mixup(self, img1, img2, label1, label2, alpha=0.2):
        """
        MixUp augmentation: linear interpolation of images and labels
        """
        lam = np.random.beta(alpha, alpha)
        mixed_img = lam * img1 + (1 - lam) * img2
        mixed_label = lam * F.one_hot(torch.tensor(label1), num_classes=self.num_classes) + \
                      (1 - lam) * F.one_hot(torch.tensor(label2), num_classes=self.num_classes)
        return mixed_img, mixed_label

    def cutmix(self, img1, img2, label1, label2):
        """
        CutMix augmentation: cut and paste regions between images
        """
        h, w = img1.shape[1], img1.shape[2]

        # Random rectangle coordinates
        cut_ratio = np.random.beta(1.0, 1.0)
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        # Create bbox
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        # Copy and paste
        img_result = img1.clone()
        img_result[:, bby1:bby2, bbx1:bbx2] = img2[:, bby1:bby2, bbx1:bbx2]

        # Adjust labels
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        mixed_label = lam * F.one_hot(torch.tensor(label1), num_classes=self.num_classes) + \
                      (1 - lam) * F.one_hot(torch.tensor(label2), num_classes=self.num_classes)

        return img_result, mixed_label

    def lighting_variation(self, img):
        """
        Apply random lighting variations to simulate different environments
        """
        if not isinstance(img, Image.Image):
            img = TF.to_pil_image(img)

        # Random brightness
        brightness_factor = np.random.uniform(0.7, 1.3)
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)

        # Random contrast
        contrast_factor = np.random.uniform(0.7, 1.3)
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)

        # Random gamma
        gamma = np.random.uniform(0.7, 1.3)
        img = TF.adjust_gamma(img, gamma)

        return TF.to_tensor(img)

    def perspective_transform(self, img):
        """
        Apply random perspective transformation to simulate different viewing angles
        """
        if not isinstance(img, Image.Image):
            img = TF.to_pil_image(img)

        width, height = img.size

        # Define the coefficients randomly
        startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
        endpoints = []

        for point in startpoints:
            displacement = np.random.randint(-width // 10, width // 10, size=2)
            endpoints.append((point[0] + displacement[0], point[1] + displacement[1]))

        return TF.to_tensor(TF.perspective(img, startpoints, endpoints, fill=0))

    def random_occlusion(self, img):
        """
        Apply random occlusions to simulate partially covered hand gestures
        """
        c, h, w = img.shape

        # Create between 1-3 random occlusion boxes
        num_boxes = np.random.randint(1, 4)
        mask = torch.ones_like(img)

        for _ in range(num_boxes):
            box_w = np.random.randint(w // 10, w // 4)
            box_h = np.random.randint(h // 10, h // 4)

            x = np.random.randint(0, w - box_w)
            y = np.random.randint(0, h - box_h)

            mask[:, y:y + box_h, x:x + box_w] = 0

        return img * mask

    def motion_blur(self, img):
        """
        Apply motion blur to simulate hand movement during capture
        """
        if not isinstance(img, Image.Image):
            img = TF.to_pil_image(img)

        # Apply motion blur in random direction
        angle = np.random.randint(0, 360)
        kernel_size = np.random.choice([5, 7, 9])

        # Create a motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel_mid = kernel_size // 2

        angle_rad = np.deg2rad(angle)
        for i in range(kernel_size):
            offset = i - kernel_mid
            x = int(round(np.cos(angle_rad) * offset))
            y = int(round(np.sin(angle_rad) * offset))

            if -kernel_mid <= x <= kernel_mid and -kernel_mid <= y <= kernel_mid:
                kernel[kernel_mid + y, kernel_mid + x] = 1

        # Ensure the kernel is not all zeros
        if np.sum(kernel) == 0:
            kernel[kernel_mid, kernel_mid] = 1

        kernel = kernel / np.sum(kernel)

        # Use try-except to handle potential errors
        try:
            img = img.filter(ImageFilter.Kernel((kernel_size, kernel_size), kernel.flatten()))
        except ValueError:
            # If there's an error, return the original image
            pass

        return TF.to_tensor(img)

    def noise_injection(self, img):
        """
        Add random noise to simulate camera noise in low light conditions
        """
        noise = torch.randn_like(img) * np.random.uniform(0.01, 0.05)
        img = img + noise
        return torch.clamp(img, 0, 1)

    def color_jitter(self, img):
        """
        Apply random color jittering to simulate different lighting and camera conditions
        """
        if not isinstance(img, Image.Image):
            img = TF.to_pil_image(img)

        jitter = transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        )

        return TF.to_tensor(jitter(img))

    def background_variation(self, img, min_val=0.0, max_val=0.3):
        """
        Apply random background intensity to simulate different backgrounds
        """
        c, h, w = img.shape

        # Create random background
        bg_value = np.random.uniform(min_val, max_val)
        background = torch.ones_like(img) * bg_value

        # Create foreground mask - assuming hand is brighter than background
        mask = (img > bg_value + 0.1).float()

        # Blend image with background
        result = img * mask + background * (1 - mask)

        return result

    def __call__(self, img1, label1, dataset=None):
        """
        Apply random advanced augmentations to the image
        """
        if np.random.rand() > self.prob:
            return img1, F.one_hot(torch.tensor(label1), num_classes=self.num_classes)

        # Choose augmentation type
        aug_type = np.random.choice(['single', 'mixup', 'cutmix'])

        if aug_type == 'single':
            # Apply single image augmentations
            aug_methods = [
                self.lighting_variation,
                self.perspective_transform,
                self.random_occlusion,
                self.motion_blur,
                self.noise_injection,
                self.color_jitter,
                self.background_variation
            ]

            # Apply 1-2 random augmentations
            num_augs = np.random.randint(1, 3)
            methods = np.random.choice(aug_methods, num_augs, replace=False)

            img_result = img1
            for method in methods:
                img_result = method(img_result)

            return img_result, F.one_hot(torch.tensor(label1), num_classes=self.num_classes)

        elif dataset is not None:
            # Get another random sample for mixup/cutmix
            idx2 = np.random.randint(len(dataset))
            img2, label2 = dataset[idx2]

            if aug_type == 'mixup':
                return self.mixup(img1, img2, label1, label2)
            else:  # cutmix
                return self.cutmix(img1, img2, label1, label2)
        else:
            # Fallback to single image augmentation
            return self.lighting_variation(img1), F.one_hot(torch.tensor(label1), num_classes=self.num_classes)


class FolderSignLanguageDataset(Dataset):
    """
    Dataset class for loading images from folders where each folder represents a class.
    """

    def __init__(self, root_dir, transform=None, augment=False, advanced_augment=False):
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        self.advanced_augment = advanced_augment

        # Get all subfolders (classes)
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Create list of (image_path, class_idx) tuples
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(class_dir, img_name), class_idx))

        print(f"Found {len(self.samples)} images across {len(self.classes)} classes")

        # Basic augmentations
        self.rotation_range = (-20, 20)  # Increased rotation range
        self.scale_range = (0.8, 1.2)  # Increased scale range
        self.brightness_range = (0.7, 1.3)
        self.contrast_range = (0.7, 1.3)

        # Advanced augmentations
        if advanced_augment:
            self.adv_aug = AdvancedAugmentation(prob=0.5, num_classes=len(self.classes))

    def __len__(self):
        return len(self.samples)

    def custom_transform(self, image):
        """
        Apply basic custom transformations to the image
        """
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
            image = TF.resize(image, (50, 50))  # Resize back to original size

        if random.random() > 0.5:
            brightness_factor = random.uniform(*self.brightness_range)
            image = TF.adjust_brightness(image, brightness_factor)

        if random.random() > 0.5:
            contrast_factor = random.uniform(*self.contrast_range)
            image = TF.adjust_contrast(image, contrast_factor)

        image = TF.to_tensor(image)
        return image

    def __getitem__(self, idx):
        """
        Get a sample from the dataset with optional augmentations
        """
        img_path, label = self.samples[idx]

        # Load image using PIL
        image = Image.open(img_path).convert('RGB')

        # Apply basic augmentation
        if self.augment:
            image = self.custom_transform(image)
        else:
            # Just convert to tensor
            image = TF.to_tensor(image)

        # Apply standard transformations
        if self.transform:
            image = self.transform(image)

        # Apply advanced augmentation if enabled
        if self.advanced_augment:
            image, one_hot_label = self.adv_aug(image, label, dataset=self)
            return image, torch.argmax(one_hot_label).item()

        return image, label

    def get_num_classes(self):
        return len(self.classes)

def create_transforms():
    """Create transformation pipeline for training data."""
    transform = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=(-20, 20),  # Increased rotation range
                translate=(0.1, 0.1),
                scale=(0.8, 1.2),   # Increased scale range
                shear=(-10, 10)     # Increased shear range
            )
        ], p=0.5),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.3))  # Increased blur
        ], p=0.3),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),  # Increased erasing area
        transforms.Resize((224, 224)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform


def create_model(num_classes):
    """Configure MobileNetV3-Large model for the task."""
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

    # Improved classifier head with more dropout for regularization
    model.classifier = nn.Sequential(
        nn.Linear(960, 1024),  # Changed from 576 to 960 for MobileNetV3-Large
        nn.Hardswish(),
        nn.Dropout(p=0.4),
        nn.Linear(1024, 512),
        nn.Hardswish(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes)
    )
    return model


def train_model(model, train_loader, device, num_epochs=30):
    """Train the model using provided data loader."""
    criterion = nn.CrossEntropyLoss()

    # Added weight decay and adjusted learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

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

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * train_correct / train_total

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')

        # Save model based on training accuracy
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'best_train_acc': best_train_acc,
                'class_to_idx': train_loader.dataset.class_to_idx  # Save class mapping for inference
            }, 'best_model_v2.pth')

    return model


def main():
    # Configuration
    data_dir = 'sign_language_dataset'  # Path to your dataset folder
    batch_size = 32
    num_epochs = 30

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataset from folder structure
    train_dataset = FolderSignLanguageDataset(
        root_dir=data_dir,
        transform=create_transforms(),
        augment=True,
        advanced_augment=True  # Enable advanced augmentations
    )

    num_classes = train_dataset.get_num_classes()
    print(f"Number of classes in dataset: {num_classes}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = create_model(num_classes)
    model = model.to(device)

    train_model(model, train_loader, device, num_epochs=num_epochs)
    print("Training completed!")


if __name__ == "__main__":
    main()