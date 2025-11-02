import torch.nn as nn
from torchvision.models import resnet18
import torch

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

def _dcgan_weights_init(m):
    cname = m.__class__.__name__
    if "Conv" in cname or "ConvTranspose" in cname:
        nn.init.normal_(m.weight, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif "BatchNorm" in cname:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif "Linear" in cname:
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.zeros_(m.bias)

class MNISTGenerator(nn.Module):
    """ Generator: z -> fake MNIST (1,28,28) """
    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # -> 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=True),     # -> 28x28
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 128, 7, 7)
        return self.net(x)


class MNISTDiscriminator(nn.Module):
    """ Discriminator: real/fake MNIST -> probability """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 128, 1),
            nn.Sigmoid()   
        )

    def forward(self, x):
        return self.classifier(self.features(x)).view(-1)


class MNISTGAN(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim
        self.generator = MNISTGenerator(z_dim)
        self.discriminator = MNISTDiscriminator()
        self.apply(_dcgan_weights_init)

    def forward(self, z):
        return self.generator(z)

# ---------- 2. get_model ----------
def get_model(model_name):
    model_name = model_name.lower()

    if model_name == "fcnn":
        model = FCNN()
    elif model_name == "cnn":
        model = CNN()
    elif model_name == "enhancedcnn":
        model = EnhancedCNN()
    elif model_name == "resnet18":
        model = resnet18(weights=None, num_classes=10)
    elif model_name == "gan":
        model = MNISTGAN(z_dim=100) 
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model