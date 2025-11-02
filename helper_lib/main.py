# helper_lib/main.py
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import get_data_loader
from helper_lib.trainer import train_model, train_gan
from helper_lib.model import get_model
from helper_lib.utils import save_model 
try:
    from helper_lib.generator import generate_samples as _gan_generate_samples
except Exception:
    _gan_generate_samples = None


# CNN 训练/评估入口（保留你当前流程）
def train_cnn_entry(
    train_dir: str = "data/train",
    test_dir: str = "data/test",
    model_name: str = "CNN",
    batch_size: int = 64,
    lr: float = 1e-3,
    epochs: int = 5,
    device: str = "cpu",
):
    device = torch.device(device)
    train_loader = get_data_loader(train_dir, batch_size=batch_size)
    test_loader = get_data_loader(test_dir, batch_size=batch_size, train=False)

    model = get_model(model_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    trained_model = train_model(model, train_loader, criterion, optimizer, device=device.type, epochs=epochs)
    evaluate_model(trained_model, test_loader, criterion)
    # 保存（沿用你现有的工具）
    save_model(trained_model, f"checkpoints/{model_name.lower()}.pt")
    return {"ckpt": f"checkpoints/{model_name.lower()}.pt"}


# MNIST DataLoader（仅供 GAN 使用）
# 保证输出已归一化到 [-1, 1]
def _get_mnist_loader(batch_size=128, train=True):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # [0,1] -> [-1,1] （配合 G 的 Tanh）
    ])
    ds = datasets.MNIST(root="./data", train=train, download=True, transform=transform(tfm) if callable(transform) else tfm)
    # 某些环境 transform 名称被覆盖，这里兼容一下：
    if hasattr(transforms, "Compose"):
        ds = datasets.MNIST(root="./data", train=train, download=True, transform=tfm)

    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)


# GAN 训练入口（MNIST）
def train_gan_entry(
    batch_size: int = 128,
    lr: float = 2e-4,
    beta1: float = 0.5,
    epochs: int = 10,
    device: str = "cpu",
    z_dim: int = 100,
):
    device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    loader = _get_mnist_loader(batch_size=batch_size, train=True)

    gan = get_model("gan")  # 在 model.py 里已经实现 MNISTGAN
    # 可选：覆盖 z_dim（如果你想改）
    if hasattr(gan, "z_dim") and gan.z_dim != z_dim:
        gan.z_dim = z_dim

    gan, history = train_gan(
        gan,
        loader,
        device=device,
        epochs=epochs,
        lr=lr,
        beta1=beta1,
        label_smooth=0.9,
        use_amp=False,
    )

    out_dir = Path("data/gan"); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "gan.pt"
    torch.save(gan.state_dict(), ckpt_path)
    return {"ckpt": str(ckpt_path), "history": history}


# GAN 采样入口
def sample_gan_entry(
    ckpt: str = "data/gan/gan.pt",
    device: str = "cpu",
    num_samples: int = 16,
    nrow: int = 4,
    out_path: str = "data/gan/samples.png",
):
    device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    if _gan_generate_samples is not None:
        return {"image": _gan_generate_samples(
            ckpt_path=ckpt,
            device=device,
            num_samples=num_samples,
            nrow=nrow,
            out_path=out_path,
            show=False
        ) or out_path}

    gan = get_model("gan")
    if Path(ckpt).exists():
        gan.load_state_dict(torch.load(ckpt, map_location=device))
    gan.to(device)
    gan.eval()

    z = torch.randn(num_samples, gan.z_dim, device=device)
    imgs = gan(z).cpu()
    imgs = (imgs + 1) / 2.0

    # 保存网格
    from torchvision.utils import save_image
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_image(imgs, out_path, nrow=nrow)
    return {"image": out_path}


if __name__ == "__main__":
    from helper_lib.evaluator import evaluate_model 
    info = train_cnn_entry(
        train_dir="data/train",
        test_dir="data/test",
        model_name="CNN",
        batch_size=64,
        lr=1e-3,
        epochs=5,
        device="cpu",
    )
    print(f"✅ CNN checkpoint saved to: {info['ckpt']}")

