import jax
import numpy as np
import torch
import torchtext
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from tqdm import tqdm
from datasets import load_dataset, DatasetDict

# Dataloaders taken directly from https://github.com/srush/annotated-s4.

def create_sin_x_dataset(n_examples=1024, batch_size=128):
    print("[*] Generating Toy Dataset: sin(x)...")

    SEQ_LENGTH, N_CLASSES, IN_DIM = 16, 8, 1
    x = np.linspace(0, 2 * np.pi, num=SEQ_LENGTH)
    y = np.digitize(np.sin(x), np.linspace(-1, 1, num=N_CLASSES))

    data = torch.Tensor(
        np.tile(
            np.expand_dims(np.expand_dims(y, -1), 0), reps=[n_examples, 1, 1]
        )
    )

    train = TensorDataset(data, data)
    test = TensorDataset(data[:1], data[:1])

    TRAIN_SIZE = len(train)

    trainloader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_mnist_dataset(batch_size=128):
    print("[*] Generating MNIST Sequence Modeling Dataset...")

    SEQ_LENGTH, N_CLASSES, IN_DIM = 784, 256, 1

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: (x.view(IN_DIM, SEQ_LENGTH).t() * 255).int()
            ),
        ]
    )

    train = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=tf
    )
    test = torchvision.datasets.MNIST(
        "./data", train=False, download=True, transform=tf
    )

    TRAIN_SIZE = len(train)

    trainloader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
    )
    testloader = torch.utils.data.DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_kmnist_dataset(batch_size=128):
    print("[*] Generating KMNIST Sequence Modeling Dataset...")

    SEQ_LENGTH, N_CLASSES, IN_DIM = 784, 256, 1

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: (x.view(IN_DIM, SEQ_LENGTH).t() * 255).int()
            ),
        ]
    )

    train = torchvision.datasets.KMNIST(
        "./data", train=True, download=True, transform=tf
    )
    test = torchvision.datasets.KMNIST(
        "./data", train=False, download=True, transform=tf
    )

    TRAIN_SIZE = len(train)

    trainloader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
    )
    testloader = torch.utils.data.DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_mnist_classification_dataset(batch_size=128):
    print("[*] Generating MNIST Classification Dataset...")

    SEQ_LENGTH, N_CLASSES, IN_DIM = 784, 10, 1
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
            transforms.Lambda(lambda x: x.view(IN_DIM, SEQ_LENGTH).t()),
        ]
    )

    train = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=tf
    )
    test = torchvision.datasets.MNIST(
        "./data", train=False, download=True, transform=tf
    )

    TRAIN_SIZE = len(train)

    trainloader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_kmnist_classification_dataset(batch_size=128):
    print("[*] Generating KMNIST Classification Dataset...")

    SEQ_LENGTH, N_CLASSES, IN_DIM = 784, 10, 1
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
            transforms.Lambda(lambda x: x.view(IN_DIM, SEQ_LENGTH).t()),
        ]
    )

    train = torchvision.datasets.KMNIST(
        "./data", train=True, download=True, transform=tf
    )
    test = torchvision.datasets.KMNIST(
        "./data", train=False, download=True, transform=tf
    )

    TRAIN_SIZE = len(train)

    trainloader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_cifar_classification_dataset(batch_size=128):
    print("[*] Generating CIFAR-10 Classification Dataset")

    SEQ_LENGTH, N_CLASSES, IN_DIM = 32 * 32, 10, 3
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
            transforms.Lambda(lambda x: x.view(IN_DIM, SEQ_LENGTH).t()),
        ]
    )

    train = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=tf
    )
    test = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=tf
    )

    TRAIN_SIZE = len(train)

    trainloader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


Datasets = {
    "sin": create_sin_x_dataset,
    "mnist": create_mnist_dataset,
    "kmnist": create_kmnist_dataset,
    "mnist-classification": create_mnist_classification_dataset,
    "kmnist-classification": create_kmnist_classification_dataset,
    "cifar-classification": create_cifar_classification_dataset,
}
