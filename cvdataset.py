import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from data_partition import partition_data


# ----------------------------
# CIFAR dataset loader
# ----------------------------
def cifar_dataset_read(dataset, base_path, batch_size, n_parties, partition, beta, skew_class):
    """
    dataset: 'cifar10' or 'cifar100'
    """

    # Data augmentation for train, standard normalization for test
    if dataset == "cifar10":
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2616))
    else:  # cifar100
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                         (0.2675, 0.2565, 0.2761))

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # ----------------------------
    # Load CIFAR dataset (with download=True!)
    # ----------------------------
    if dataset == "cifar10":
        train_dataset = CIFAR10(root=base_path, train=True, download=True, transform=train_transform)
        test_dataset = CIFAR10(root=base_path, train=False, download=True, transform=test_transform)
    else:
        train_dataset = CIFAR100(root=base_path, train=True, download=True, transform=train_transform)
        test_dataset = CIFAR100(root=base_path, train=False, download=True, transform=test_transform)

    # ----------------------------
    # Partition data
    # ----------------------------
    X_train = np.array(train_dataset.data)
    y_train = np.array(train_dataset.targets)

    n_train = len(y_train)

    # Call data partition function
    net_dataidx_map, client_num_samples, traindata_cls_counts, data_distributions = \
        partition_data(partition, n_train, n_parties, y_train, beta, skew_class)

    # ----------------------------
    # Create per-client dataloaders
    # ----------------------------
    train_loaders = []
    for i in range(n_parties):
        idx = net_dataidx_map[i]
        subset = Subset(train_dataset, idx)
        train_loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=False))

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=False)

    return train_loaders, test_loader, client_num_samples, traindata_cls_counts, data_distributions


# ----------------------------
# Fashion-MNIST loader (unchanged)
# ----------------------------
from torchvision.datasets import FashionMNIST


def fashionmnist_dataset_read(dataset, base_path, batch_size, n_parties, partition, beta, skew_class):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = FashionMNIST(root=base_path, train=True, download=True, transform=transform)
    test_dataset = FashionMNIST(root=base_path, train=False, download=True, transform=transform)

    X_train = train_dataset.data.numpy()
    y_train = train_dataset.targets.numpy()

    n_train = len(y_train)

    net_dataidx_map, client_num_samples, traindata_cls_counts, data_distributions = \
        partition_data(partition, n_train, n_parties, y_train, beta, skew_class)

    train_loaders = []
    for i in range(n_parties):
        idx = net_dataidx_map[i]
        subset = Subset(train_dataset, idx)
        train_loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=False))

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=False)

    return train_loaders, test_loader, client_num_samples, traindata_cls_counts, data_distributions
