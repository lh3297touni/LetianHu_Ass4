import torch
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from pathlib import Path
from .model import get_model


@torch.no_grad()
def generate_samples(
    ckpt_path: str = "data/gan/gan.pt",
    device: str = "cpu",
    num_samples: int = 16,
    nrow: int = 4,
    out_path: str = "data/gan/samples.png",
    show: bool = True,
):
    device = torch.device(device)

    # 1) 加载模型
    gan = get_model("gan")  # 会创建 MNISTGAN(z_dim=100)
    if Path(ckpt_path).exists():
        gan.load_state_dict(torch.load(ckpt_path, map_location=device))
    gan.to(device)
    gan.eval()

    # 2) 随机噪声 z
    z = torch.randn(num_samples, gan.z_dim, device=device)

    # 3) 生成假图像 (-1,1)
    imgs = gan(z).cpu()

    # 4) 反归一化到 (0,1)
    imgs = (imgs + 1) / 2.0

    # 5) 保存图像文件
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_image(imgs, out_path, nrow=nrow)
        print(f"✅ 图像已保存到: {out_path}")

    # 6) 显示网格
    if show:
        grid = make_grid(imgs, nrow=nrow, padding=2)
        plt.figure(figsize=(6,6))
        plt.axis("off")
        plt.title("Generated Samples")
        plt.imshow(grid.permute(1, 2, 0), cmap="gray")
        plt.show()

    return imgs
