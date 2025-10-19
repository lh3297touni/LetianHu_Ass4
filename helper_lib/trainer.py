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
