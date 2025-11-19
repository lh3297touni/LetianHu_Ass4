import torch
from torchvision import datasets, transforms

def get_data_loader(data_dir, batch_size=32, train=True):
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  
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


#  CIFAR-10 Loader （Energy Model / Diffusion）
def get_cifar10_loaders(batch_size=128, num_workers=2, root="./data"):
    """
    CIFAR-10 Loader，用于 EBM 与 Diffusion 模型。
    输出 train_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test_ds  = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader
