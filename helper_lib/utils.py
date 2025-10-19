# helper_lib/utils.py
import os
import torch

def get_device():
    """
    自动选择可用设备：
    - 优先使用 CUDA
    - 再使用 Apple MPS（适用于 M1/M2）
    - 否则使用 CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using GPU (CUDA)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("⚙️ Using CPU")
    return device


def save_model(model, path="model.pth"):
    """
    保存模型参数到指定路径。
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"💾 Model saved to: {path}")


def load_model(model, path, device="cpu"):
    """
    从文件加载模型参数。
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"📂 Model loaded from: {path}")
    return model


def print_progress(epoch, total_epochs, loss):
    """
    打印训练进度信息。
    """
    print(f"Epoch [{epoch+1}/{total_epochs}] - Loss: {loss:.4f}")
