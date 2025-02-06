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
    Aim: This class focuses on handling data loading, preprocessing, and augmentation of sign language images.

    Parameters:
    - data: Raw pixel data of images
    - labels: Corresponding labels for the images
    - transform: Transformation pipeline to be applied to images
    - augment: Whether to apply additional data augmentations
    """

    def __init__(self, data, labels, transform=None, augment=False):
        # Create a mapping of text labels to numeric indices
        self.data = data
        self.label_map = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        self.labels = np.array([self.label_map[label] for label in labels])
        self.transform = transform
        self.augment = augment
        self.num_classes = len(self.label_map)

        # Here is about defining ranges for data augmentation parameters
        self.rotation_range = (-15, 15)  # Random rotation between -15 and +15 degrees
        self.scale_range = (0.9, 1.1)  # Random scaling between 90% and 110% of original size
        self.brightness_range = (0.8, 1.2)  # Adjust brightness between 80% and 120%
        self.contrast_range = (0.8, 1.2)  # Adjust contrast between 80% and 120%

    def __len__(self):
        return len(self.labels)

    def custom_transform(self, image):
        """
        Aim: Apply the data augmentation to images.

        parameter: the image needs to be augmented

        return: Augmented image
        """
        if not self.augment:
            return image

        if isinstance(image, torch.Tensor):
            """
            Aim: Convert torch tensor to PIL Image 
            
            More explanation: 
            torch.Tensor is about matrices, while converting them to PIL images allows us for better 
            and more intuitive image processing data augmentation
            """
            image = TF.to_pil_image(image)

        # Random rotation with 50% probability. Each image has a 50% chance to be applied the data augmentation.
        if random.random() > 0.5:
            angle = random.uniform(*self.rotation_range)
            image = TF.rotate(image, angle)

        # Random scale with 50% probability
        if random.random() > 0.5:
            scale = random.uniform(*self.scale_range)
            new_size = tuple(int(s * scale) for s in image.size)
            image = TF.resize(image, new_size)
            # Resize back to original size to maintain consistency
            image = TF.resize(image, image.size)

        # Random brightness with 50% probability
        if random.random() > 0.5:
            brightness_factor = random.uniform(*self.brightness_range)
            image = TF.adjust_brightness(image, brightness_factor)

        # Random contrast with 50% probability
        if random.random() > 0.5:
            contrast_factor = random.uniform(*self.contrast_range)
            image = TF.adjust_contrast(image, contrast_factor)

        # Convert back to tensor
        image = TF.to_tensor(image)
        return image

    def __getitem__(self, idx):
        """
        Aim: Fetch and preprocess a single data item.

        Parameter: idx: Index of the data

        Returns: preprocessed image tensor and label
        """


        """
        Aim: Reshape the flat array into 28x28 image and normalize to [0,1]
        
        More explanation: 
        1. Flay array (or called '1D array'): The training data is stored in a .cvs file as a 1D array, with each row 
        containing 784 pixel values (28*28 = 784). You can open that file to see it. However, images are 2D array, right, 
        and this 1D array needs to be reshaped into a 28x28 square matrix to correctly represent the image.
        
        2. Why 28x28: Because 28x28 is the standard image size of the 'Sign Language MNIST' dataset
        """
        image = self.data[idx].reshape(28, 28).astype(np.float32)
        image = image / 255.0



        """
        Aim: Convert to 3 channels
        
        More explanation: 
        1. MobileNetV3-Small is a lightweight CNNs model which trained on the ImageNet dataset.
        
        2. We can fine-tune this model for our task because our input is of the same type (images), 
        and the ImageNet dataset has 2.6M images, while we only have 30K images, so we can use transfer learning.
        
        3. Back to the question that why need to convert to 3 channels here: Because MobileNetV3-Small's first layer 
        needs 3-channel (RGB) inputs. We have to keep the input type the same.
        """
        image = np.stack([image] * 3, axis=0)
        image = torch.FloatTensor(image)

        # Apply data augmentation
        image = self.custom_transform(image)


        """
        Aim: Apply standard transforms (resize, normalize, etc.)
        
        More explanation:
        The MobileNetV3 model is trained on the ImageNet dataset, which uses an input size of 224x224.
        Our input size is 28x28.
        """
        if self.transform:
            image = self.transform(image)


        label = int(self.labels[idx])
        return image, label

    def get_num_classes(self):
        return self.num_classes


def create_transforms(is_training=True):
    """
    Aim: Create transformation pipelines for training and validation data.
    I used 80% of the 'Sign Language MNIST' dataset as a training set and 20% as a validation set.
    Or you don’t need a validation set. I think both situations are fine.

    Parameters: is_training (bool): Whether to create transforms for training or validation

    Return: Composition of image transformations
    """

    if is_training:
        """
        Aim: More aggressive transformations for training data
        
        Here, I made the data augmentation again to add more challenges during training, lol. 
        However, you can only add the data augmentation once if you want. :)
        """
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
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.2)) # Add blur
            ], p=0.3),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)), # Randomly erase small patches
            transforms.Resize((224, 224)), # Resize for MobileNetV3 model
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], # ImageNet normalization values
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        # Simple transforms for validation data
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    return transform


def load_and_prepare_data(data_path):
    """
    Aim: Load and split the dataset into training and validation sets.

    Parameters: data_path: Path to the training set .cvs file

    Return: Training and validation sets and labels
    """
    data = pd.read_csv(data_path)
    unique_labels = sorted(data['label'].unique())
    print(f"Unique labels: {unique_labels}")
    print(f"Number of classes: {len(unique_labels)}")

    # Separate features and labels
    X = data.drop('label', axis=1).values # Remove 'label' from the file and only keep all image pixel data
    y = data['label'].values # only keep 'label'


    """
    Aim: Split data with stratification to maintain class distribution
    
    More explanation:
    1. test_size=0.2: means that 80% of the data is used for training and 20% is used for validation. 
    (I don't know why I named it "test_size", please ignore it, lol)
    
    2. stratify=y: I think this is important. 
    Emmm, for example, in the 'Sign Language MNIST' set: 'A' has 30%, 'B' has 20% and 'C' has 50%.
    After using stratify=y, the training set and the validation set will keep the same ratio.
    But if you don't want to use the val set you can skip it.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_val, y_train, y_val



# IMPORTANT!!! Fine-tune part
def create_model(num_classes):
    """
    Aim: Load and modify the MobileNetV3-Small model for our task.

    Parameters: Number of output classes

    Returns: Configured MobileNetV3-Small model
    """

    # Load pretrained MobileNetV3-Small model
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

    # Replace the classifier layer with a custom one
    model.classifier = nn.Sequential( # Create a sequential neural network layer

        # First layer: fully connected layer, expanding the feature dimension from 576 to 1024
        nn.Linear(576, 1024),

        # Second layer: activation function layer
        nn.Hardswish(),

        # Third layer: Dropout layer
        nn.Dropout(p=0.3),  # Increased dropout for better regularization

        # Fourth layer: final classification layer
        nn.Linear(1024, num_classes)
    )
    """
    More explanation on the above:
    1. First layer: The Linear layer expands the feature dimension from 576 to 1024 to enhance the model's capacity for 
    learning more nuanced features.
    
    2. Second layer: Hardswish is the recommended activation function in MobileNetV3 paper as it offers better 
    performance compared to ReLU. (I have always used ReLU before, this is my first time to use Hardswish)
    
    3. Third layer: Dropout prevents overfitting by randomly disabling a portion (e.g., 30%) of neurons during training, 
    forcing the network to learn robust features. 
    
    4. Fourth layer: The final layer maps the 1024-dimensional features to the number of sign language classes 
    (e.g., 24 classes for sign language).
    """
    return model


# Ah, I am tired here...I am gonna to play the 'Black Myth: WuKong' game, haha!!!


def train_model(model, train_loader, val_loader, device, num_epochs=30):
    """
    Aim: Train the model using the provided data loaders.

    Parameters:
        model: The neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on (CPU or GPU)
        num_epochs: Number of training epochs

    Returns: Trained model
    """
    criterion = nn.CrossEntropyLoss()  # This is the loss function for multi-classification problems

    # Adam optimizer: set the learning rate to 0.001 and add weight_decay to prevent overfitting
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    """
    Learning rate scheduler: 
    This scheduler will reduce the learning rate to help the model find a better local optimal solution.
    """
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=3,  # If there is no improvement for 3 consecutive epochs, reduce the learning rate
        factor=0.5,  # The learning rate is reduced to half of its original value.
        min_lr=1e-6  # Minimum learning rate
    )

    best_train_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()  # Set to training mode (enable dropout, etc.)
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):

            # Move data to the specified device (GPU/CPU)
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero gradient
            outputs = model(images)  # Forward Propagation
            loss = criterion(outputs, labels)  # Calculating Loss
            loss.backward()  # Back Propagation

            # Gradient clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update Parameters
            optimizer.step()

            train_loss += loss.item()

            # Calculate training accuracy
            _, predicted = outputs.max(1)  # Get the classification with the highest probability
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()  # Count the number of correct predictions

        train_acc = 100.0 * train_correct / train_total

        # Validation phase
        model.eval()  # Set to val mode (disable dropout, etc.)
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():  # Do not calculate gradients because there is no need for val phase
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Calculation val accuracy
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')

        scheduler.step(val_acc)

        # Save the model with the highest training accuracy
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,  # Save the training set accuracy
                'best_train_acc': best_train_acc  # You can also save the best training set accuracy
            }, 'best_train_model.pth')  # This is the trained model file which will be saved to your directory

    return model


def main():

    # If there is a GPU, use the GPU, if not, use the CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and prepare data
    X_train, X_val, y_train, y_val = load_and_prepare_data('sign_mnist_train.csv')

    # Create datasets with augmentation for training
    train_dataset = SignLanguageDataset(
        X_train, y_train,
        transform=create_transforms(is_training=True),
        augment=True
    )
    val_dataset = SignLanguageDataset(
        X_val, y_val,
        transform=create_transforms(is_training=False),
        augment=False # False: not use data augmentation
    )

    # Get the number of classifications
    num_classes = train_dataset.get_num_classes()
    print(f"Number of classes in dataset: {num_classes}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,  # 32 images are processed per batch
        shuffle=True,  # Ensure that the training data order of each epoch is different to increase randomness
        num_workers=4,  # Loading data using 4 processes
        pin_memory=True  # Data is loaded directly into GPU memory (if using GPU)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,  # The vali set does not need to be shuffled
        num_workers=4,
        pin_memory=True
    )

    model = create_model(num_classes)  # Create model
    model = model.to(device)  # Move the model to the specified device (GPU/CPU)

    train_model(model, train_loader, val_loader, device)  # Start training
    print("Training completed!")


if __name__ == "__main__":
    main()