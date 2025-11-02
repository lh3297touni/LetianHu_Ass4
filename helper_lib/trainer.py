import torch

def train_model(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    # 把模型放到指定设备（CPU 或 GPU）
    model.to(device)

    # 训练循环
    for epoch in range(epochs):
        model.train()  # 切换到训练模式
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            # 把数据移到 device 上
            inputs, targets = inputs.to(device), targets.to(device)

            # 1️⃣ 清空上一次的梯度
            optimizer.zero_grad()

            # 2️⃣ 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 3️⃣ 反向传播
            loss.backward()

            # 4️⃣ 更新权重
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    print("✅ Training complete.")
    return model

# ====== GAN 训练（MNIST, 按作业要求） ======
from torch import nn
from torch.optim import Adam

@torch.no_grad()
def _gan_d_acc(d_out_real, d_out_fake):
    real_ok = (d_out_real >= 0.5).float().mean().item()
    fake_ok = (d_out_fake < 0.5).float().mean().item()
    return (real_ok + fake_ok) / 2.0

def train_gan(
    model,
    data_loader,
    device: str = "cpu",
    epochs: int = 10,
    lr: float = 2e-4,
    beta1: float = 0.5,
    label_smooth: float = 0.9,
    use_amp: bool = False,
):
    assert hasattr(model, "generator") and hasattr(model, "discriminator"), "Expect MNISTGAN"
    device = torch.device(device)
    model.to(device)
    G, D = model.generator, model.discriminator

    bce = nn.BCELoss()
    optD = Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
    optG = Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = {"d_loss": [], "g_loss": [], "d_acc": []}

    for ep in range(1, epochs + 1):
        d_epoch = g_epoch = acc_epoch = 0.0
        n_batches = 0

        for real, _ in data_loader:
            n = real.size(0)
            real = real.to(device)

            # ---- 1) 训练判别器 D ----
            D.train(); G.train()
            optD.zero_grad(set_to_none=True)

            # 标签：真实=1（平滑到 label_smooth），伪造=0
            y_real = torch.full((n,), label_smooth, device=device)
            y_fake = torch.zeros(n, device=device)

            # 生成假样本（不让 G 反传进 D 这一步）
            z = torch.randn(n, model.z_dim, device=device)
            with torch.no_grad():
                fake = G(z)

            with torch.cuda.amp.autocast(enabled=use_amp):
                d_out_real = D(real)
                d_out_fake = D(fake)
                d_loss = bce(d_out_real, y_real) + bce(d_out_fake, y_fake)

            scaler.scale(d_loss).backward()
            scaler.step(optD)

            # ---- 2) 训练生成器 G ----
            optG.zero_grad(set_to_none=True)
            z = torch.randn(n, model.z_dim, device=device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                gen = G(z)
                # 让 D 将假样本判为真
                g_loss = bce(D(gen), torch.ones(n, device=device))

            scaler.scale(g_loss).backward()
            scaler.step(optG)
            scaler.update()

            with torch.no_grad():
                acc = _gan_d_acc(d_out_real, d_out_fake)

            d_epoch += float(d_loss)
            g_epoch += float(g_loss)
            acc_epoch += acc
            n_batches += 1

        d_avg = d_epoch / n_batches
        g_avg = g_epoch / n_batches
        acc_avg = acc_epoch / n_batches
        history["d_loss"].append(d_avg)
        history["g_loss"].append(g_avg)
        history["d_acc"].append(acc_avg)

        print(f"[Epoch {ep:03d}/{epochs}] D_loss: {d_avg:.4f}  G_loss: {g_avg:.4f}  D_acc: {acc_avg*100:.1f}%")

    print("✅ GAN training complete.")
    return model, history
