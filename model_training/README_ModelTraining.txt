Sign Language Recognition Model

1. Training Guide

1.1 Prerequisites
	- Conda installed
	- NVIDIA GPU with CUDA support

1.1.1 Setup & Training
	- Create environment:
		```bash
		conda create -n model_training python=3.8
		conda activate model_training
		pip install -r requirements.txt
		```

	- Install PyTorch with CUDA
    		a. Via https://pytorch.org/get-started/previous-versions/
    		b. Or run: 
    			```bash
    			conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
			```

	- Train model:
		```bash
		python model_training_clean.py
		```
	(Note: model_training_comments.py contains detailed comments with the same code)


	- Training output:
		'best_train_model.pth': Saved model with best training accuracy




2. Using the Trained Mode (This is just an example, you can use the model in your own way)

2.1 Here's how to load and use the trained model:

```python
import torch
from torchvision.models import mobilenet_v3_small
import torch.nn as nn

def load_model(model_path, num_classes):
    # Create model architecture
    model = mobilenet_v3_small()
    model.classifier = nn.Sequential(
        nn.Linear(576, 1024),
        nn.Hardswish(),
        nn.Dropout(p=0.3),
        nn.Linear(1024, num_classes)
    )
    
    # Load trained weights
    checkpoint = torch.load('best_train_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

# Example usage
model = load_model('best_train_model.pth', num_classes=24)
```


2.2 For inference:

```python
def predict(model, image):
    # Preprocess image (ensure same preprocessing as training)
    # Make prediction
    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1)
    return prediction
```











