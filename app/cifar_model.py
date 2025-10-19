import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# CNN architecture
class SimpleCNN(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)     
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)     
        self.fc1 = nn.Linear(32 * 16 * 16, 100)               
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))   
        x = self.fc2(x)           
        return x


# classifier
class CifarClassifier:
    def __init__(self, weight_path="data/cifar_simplecnn.pt"):
        self.model = SimpleCNN()
        self.weight_path = weight_path
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        try:
            state = torch.load(self.weight_path, map_location="cpu")
            self.model.load_state_dict(state)
            self.model.eval()
            print("Loaded pre-trained CIFAR10 model.")
        except Exception:
            print("No trained model found, please train first.")

    def predict(self, img: Image.Image):
        x = self.transform(img.convert("RGB")).unsqueeze(0) 
        with torch.no_grad():
            logits = self.model(x)
            prob = torch.softmax(logits, dim=1).squeeze(0)  
            idx = int(prob.argmax().item())
        return idx, prob.tolist()

    def save(self):
        torch.save(self.model.state_dict(), self.weight_path)
