import torch
from torchvision import datasets, transforms

def get_data_loader(data_dir, batch_size=32, train=True):
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1,1]区间
    ])

    # Load dataset (示例使用MNIST)
    dataset = datasets.MNIST(
        root=data_dir,
        train=train,
        transform=transform,
        download=True
    )

    # Create DataLoader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train  # 训练集打乱，验证/测试不打乱
    )

    return loader
