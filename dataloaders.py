import jax
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from tqdm import tqdm


# ### $sin(x)$
# **Task**: Overfit to a 8-bit quantized sin(x) from 0 - 2*Pi -- sampled 360 times.
#
#  @Note: The Feed-Forward model won't necessarily be able to fit this data (optimization is hard)
#  As a sanity check, you can try running with N_CLASSES = 2 (-1, 1) and d_model = 1...
#  this is the simplest "majority rule" experiment => gets 100% test accuracy.
#
#  @Note: RNN & S4 *should* fit this perfectly... but needs to be verified.
def create_sin_x_dataset(n_examples=1024, batch_size=128):
    print("[*] Generating Toy Dataset: sin(x)...")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 16, 8, 1
    x = np.linspace(0, 2 * np.pi, num=SEQ_LENGTH)
    y = np.digitize(np.sin(x), np.linspace(-1, 1, num=N_CLASSES))

    # Tile this `n_examples` times...
    data = torch.Tensor(
        np.tile(
            np.expand_dims(np.expand_dims(y, -1), 0), reps=[n_examples, 1, 1]
        )
    )

    # Build Datasets -- Two entries to match (inputs, targets) structure
    train = TensorDataset(data, data)
    test = TensorDataset(data[:1], data[:1])

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


# ### $sin(ax + b)$
# **Task**: Fit arbitrary 8-bit quantized functions of the form sin(ax + b) from 0 - 2*Pi -- sampled 360 times.
#
# In this dataset, `a` controls amplitude and `b` controls phase and are sampled uniformly at random in prespecified
# intervals.
def create_sin_ax_b_dataset(n_examples=20000, batch_size=128):
    print("[*] Generating sin(ax + b) Dataset...")

    # Constants – `a` sampled uniform from [1, 10], `b` sampled uniform [0, 5]
    SEQ_LENGTH, N_CLASSES, IN_DIM, A_MAX, B_MAX = 16000, 8, 1, 10, 5
    train_data, test_data = [], []
    data_key = jax.random.PRNGKey(21)

    # Loop through `n_examples` and generate data
    print(f"\t=>> Generating {n_examples} Training Examples...")
    x = np.linspace(0, 2 * np.pi, num=SEQ_LENGTH)
    for _ in tqdm(range(n_examples)):
        data_key, a_rng, b_rng = jax.random.split(data_key, num=3)

        # Compute a, b
        a, b = jax.random.uniform(
            a_rng, minval=1.0, maxval=A_MAX
        ), jax.random.uniform(b_rng, maxval=B_MAX)
        train_data.append(
            np.digitize(np.sin(a * x + b), np.linspace(-1, 1, num=N_CLASSES))
        )

    # Generate 1 Batch of Test Examples
    print(f"\t=>> Generating {batch_size} Test Examples...")
    for _ in tqdm(range(batch_size)):
        data_key, a_rng, b_rng = jax.random.split(data_key, num=3)

        # Compute a, b
        a, b = jax.random.uniform(
            a_rng, minval=1.0, maxval=A_MAX
        ), jax.random.uniform(b_rng, maxval=B_MAX)
        test_data.append(
            np.digitize(np.sin(a * x + b), np.linspace(-1, 1, num=N_CLASSES))
        )

    # Build Datasets - Two entries to match (inputs, targets) structure
    train_data = torch.Tensor(np.expand_dims(np.array(train_data), -1))
    test_data = torch.Tensor(np.expand_dims(np.array(test_data), -1))
    train = TensorDataset(train_data, train_data)
    test = TensorDataset(test_data, test_data)

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


# ### MNIST Sequence Modeling
# **Task**: Predict next pixel value given history, in an autoregressive fashion (784 pixels x 256 values).
#
def create_mnist_dataset(batch_size=128):
    print("[*] Generating MNIST Sequence Modeling Dataset...")

    # Constants
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

    # Return data loaders, with the provided batch size
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

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


def create_kmnist_dataset(batch_size=128):
    print("[*] Generating KMNIST Sequence Modeling Dataset...")

    # Constants
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

    # Return data loaders, with the provided batch size
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

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


# ### MNIST Classification
# **Task**: Predict MNIST class given sequence model over pixels (784 pixels => 10 classes).
def create_mnist_classification_dataset(batch_size=128):
    print("[*] Generating MNIST Classification Dataset...")

    # Constants
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

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


def create_kmnist_classification_dataset(batch_size=128):
    print("[*] Generating KMNIST Classification Dataset...")

    # Constants
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

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


# ### CIFAR-10 Classification
# **Task**: Predict CIFAR-10 class given sequence model over pixels (32 x 32 x 3 RGB image => 10 classes).
def create_cifar_classification_dataset(batch_size=128):
    print("[*] Generating CIFAR-10 Classification Dataset")

    # Constants
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

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM



Datasets = {
    "sin": create_sin_x_dataset,
    "sin_noise": create_sin_ax_b_dataset,
    "mnist": create_mnist_dataset,
    "kmnist": create_kmnist_dataset,
    "mnist-classification": create_mnist_classification_dataset,
    "kmnist-classification": create_kmnist_classification_dataset,
    "cifar-classification": create_cifar_classification_dataset,
}
