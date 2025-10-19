import torch.nn as nn
from torchvision.models import resnet18

# ---------- 1. 定义几个模型 ----------
class FCNN(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        return self.net(x)


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Linear(64 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class EnhancedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Dropout(0.25)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ---------- 2. get_model ----------
def get_model(model_name):
    # TODO: define and return the appropriate model_name - one of: FCNN, CNN, EnhancedCNN
    if model_name.lower() == "fcnn":
        model = FCNN()
    elif model_name.lower() == "cnn":
        model = CNN()
    elif model_name.lower() == "enhancedcnn":
        model = EnhancedCNN()
    elif model_name.lower() == "resnet18":
        model = resnet18(weights=None, num_classes=10)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model
