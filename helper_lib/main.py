from helper_lib.data_loader import get_data_loader
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model
from helper_lib.model import get_model
from helper_lib.utils import save_model
import torch.nn as nn
import torch.optim as optim
train_loader = get_data_loader('data/train', batch_size=64)
test_loader = get_data_loader('data/test', batch_size=64, train=False)
model = get_model("CNN")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
trained_model = train_model(model, train_loader, criterion, optimizer, epochs=5)
evaluate_model(trained_model, test_loader, criterion)

