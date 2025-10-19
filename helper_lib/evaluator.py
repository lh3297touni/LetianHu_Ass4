import torch

def evaluate_model(model, data_loader, criterion, device='cpu'):
    model.to(device)
    model.eval()  # 切换到评估模式

    total_loss = 0.0
    correct = 0
    total = 0

    # 不需要梯度计算，加快推理速度
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # 预测类别
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * correct / total

    print(f"Evaluation - Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy
