import torch
import torch.nn as nn
from pathlib import Path
from torchvision.utils import save_image

from .data_loader import get_cifar10_loaders


class CIFAREnergyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(256 * 4 * 4, 1)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(x.size(0), -1)
        energy = self.fc(h)
        return energy  # (B, 1)
    

def train_energy(
    batch_size: int = 128,
    lr: float = 1e-4,
    epochs: int = 5,
    device: str = "cpu",
    ckpt_path: str = "data/energy/energy.pt",
    data_root: str = "./data",
):

    device = torch.device(device)
    train_loader, _ = get_cifar10_loaders(batch_size=batch_size, root=data_root)

    model = CIFAREnergyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ckpt_file = Path(ckpt_path)
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x, _ in train_loader:
            x = x.to(device)

            e_pos = model(x).mean()

            noise = torch.randn_like(x) * 0.3
            x_neg = (x + noise).clamp(-1, 1)
            e_neg = model(x_neg).mean()

            loss = e_pos - e_neg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"[EBM] Epoch {ep}/{epochs} - loss={avg_loss:.4f}")

    torch.save(model.state_dict(), ckpt_file)
    print(f"✅ Energy model saved to: {ckpt_file}")
    return {"ckpt": str(ckpt_file), "loss": avg_loss}


def sample_energy(
    device: str = "cpu",
    ckpt_path: str = "data/energy/energy.pt",
    steps: int = 60,
    step_size: float = 0.1,
    num_samples: int = 16,
    out_path: str = "data/energy/samples.png",
):

    device = torch.device(device)

    ckpt_file = Path(ckpt_path)
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Energy checkpoint not found: {ckpt_file.resolve()}")
    if ckpt_file.stat().st_size < 1024:
        raise RuntimeError(
            f"Energy checkpoint file too small / corrupted: {ckpt_file.resolve()}"
        )

    model = CIFAREnergyModel().to(device)
    state = torch.load(str(ckpt_file), map_location=device)
    model.load_state_dict(state)
    model.eval()

    x = torch.randn(num_samples, 3, 32, 32, device=device, requires_grad=True)

    for i in range(steps):
        if x.grad is not None:
            x.grad.zero_()

        energy = model(x).sum()
        (grad,) = torch.autograd.grad(energy, x, create_graph=False)

        noise = torch.randn_like(x)
        x = x - 0.5 * step_size * grad + torch.sqrt(torch.tensor(step_size, device=device)) * noise
        x = x.clamp(-1, 1)
        x.requires_grad_(True)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    imgs = (x.detach().cpu().clamp(-1, 1) + 1) / 2.0
    save_image(imgs, out_file, nrow=4)

    print(f"✅ Energy samples saved to: {out_file}")
    return {"image": str(out_file)}