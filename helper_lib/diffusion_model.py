import torch
import torch.nn as nn
from torchvision.utils import save_image
from pathlib import Path
from .data_loader import get_cifar10_loaders


def make_beta_schedule(T=1000, start=1e-4, end=0.02):
    return torch.linspace(start, end, T)


class SimpleUNet(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(3, ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(ch, ch * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch * 2, ch * 2, 3, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(ch * 2, ch * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ch, 3, 4, 2, 1),
        )

    def forward(self, x, t_embed):
        h1 = self.down1(x)
        h2 = self.down2(h1)
        h = self.mid(h2)
        h = self.up1(h)
        h = self.up2(h)
        return h


def train_diffusion(
    batch_size=128,
    lr=2e-4,
    epochs=5,
    T=1000,
    device="cpu",
    ckpt_path="data/diffusion/diffusion.pt",
):
    device = torch.device(device)
    train_loader, _ = get_cifar10_loaders(batch_size=batch_size)

    betas = make_beta_schedule(T).to(device)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    model = SimpleUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    ckpt_file = Path(ckpt_path)
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(epochs):
        model.train()
        total_loss = 0.0

        for x0, _ in train_loader:
            x0 = x0.to(device)

            # 随机时间步
            t = torch.randint(0, T, (x0.size(0),), device=device)
            a_bar_t = alphas_bar[t].view(-1, 1, 1, 1)

            # 前向加噪
            noise = torch.randn_like(x0)
            x_t = torch.sqrt(a_bar_t) * x0 + torch.sqrt(1 - a_bar_t) * noise

            t_embed = torch.zeros(x0.size(0), 1, device=device)
            pred_noise = model(x_t, t_embed)

            loss = torch.mean((pred_noise - noise) ** 2)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"[DDPM] epoch {ep+1}/{epochs} - loss={avg_loss:.4f}")

    torch.save(
        {
            "model": model.state_dict(),
            "betas": betas.cpu(),
        },
        ckpt_file,
    )

    print(f"✅ Diffusion model saved to: {ckpt_file}")
    return {"ckpt": str(ckpt_file), "loss": avg_loss}


@torch.no_grad()
def sample_diffusion(
    num_samples=16,
    T=1000,
    device="cpu",
    ckpt_path="data/diffusion/diffusion.pt",
    out_path="data/diffusion/samples.png",
):
    device = torch.device(device)

    ckpt_file = Path(ckpt_path)
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Diffusion checkpoint not found: {ckpt_file.resolve()}")
    if ckpt_file.stat().st_size < 1024:
        raise RuntimeError(
            f"Diffusion checkpoint file too small / corrupted: {ckpt_file.resolve()}"
        )

    ckpt = torch.load(str(ckpt_file), map_location=device)
    betas = ckpt["betas"].to(device)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    model = SimpleUNet().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    x_t = torch.randn(num_samples, 3, 32, 32, device=device)

    for t in reversed(range(T)):
        t_embed = torch.zeros(num_samples, 1, device=device)
        eps = model(x_t, t_embed)

        a_t = alphas[t]
        a_bar_t = alphas_bar[t]
        beta_t = betas[t]

        if t > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)

        x_t = (1.0 / torch.sqrt(a_t)) * (
            x_t - (beta_t / torch.sqrt(1 - a_bar_t)) * eps
        ) + torch.sqrt(beta_t) * noise

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    imgs = (x_t.clamp(-1, 1) + 1) / 2.0
    save_image(imgs, out_file, nrow=4)

    print(f"✅ Diffusion samples saved to: {out_file}")
    return {"image": str(out_file)}