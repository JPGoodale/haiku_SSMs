import jax
import jax.numpy as jnp
import haiku as hk
import optax
import torch
from functools import partial
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloaders import Datasets
from s4 import S4, S4Stack
from s5 import S5, S5Stack, S5Classifier
from typing import Tuple, NamedTuple, MutableMapping, Any, Optional


_Metrics = MutableMapping[str, Any]


STATE_SIZE: int = 128
D_MODEL: int = 64
N_LAYERS: int = 6
N_BLOCKS: int = 8
EPOCHS: int = 100
BATCH_SIZE: int = 128
DROPOUT_RATE: float = 0.5
LEARNING_RATE: float = 0.005
WEIGHT_DECAY: float = 0.01
BASIS_MEASURE: str = 'legs'
DATASET = 'mnist-classification'
SEED = 42


class Dataset(NamedTuple):
    trainloader: DataLoader
    testloader: DataLoader
    n_classes: int
    seq_length: int
    d_input: int
    train_size: int
    classification: bool


def create_dataset(dataset: str, batch_size: int) -> Dataset:
    classification = 'classification' in dataset
    dataset_init = Datasets[dataset]
    trainloader, testloader, n_classes, seq_length, d_input, train_size = dataset_init(batch_size)
    return Dataset(trainloader, testloader, n_classes, seq_length, d_input, train_size, classification)


class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState
    rng_key: jnp.ndarray


def create_optimizer(dataset: Dataset, warmup_end: int = 1) -> optax.GradientTransformation:
    steps_per_epoch = int(dataset.train_size / BATCH_SIZE)
    decay_steps = steps_per_epoch * EPOCHS - (steps_per_epoch * warmup_end)
    lr_schedule = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=decay_steps)
    optimizer = optax.adamw(lr_schedule, weight_decay=WEIGHT_DECAY)
    return optimizer


@partial(jnp.vectorize, signature="(c),()->()")
def cross_entropy_loss(logits, label) -> jnp.ndarray:
    one_hot_label = jax.nn.one_hot(label, num_classes=logits.shape[0])
    return -jnp.sum(one_hot_label * logits)


@partial(jnp.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    return jnp.argmax(logits) == label


@partial(jax.jit, static_argnums=(3, 4, 5))
def update(
        state: TrainingState,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        optimizer: optax.GradientTransformation,
        model: hk.transform,
        classification: bool = False
) -> Tuple[TrainingState, _Metrics]:

    rng_key, next_rng_key = jax.random.split(state.rng_key)

    def loss_fn(params):
        logits = model.apply(params, rng_key, inputs)
        _loss = jnp.mean(cross_entropy_loss(logits, targets))
        _accuracy = jnp.mean(compute_accuracy(logits, targets))
        return _loss, _accuracy

    if not classification:
        targets = inputs[:, :, 0]

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, accuracy), gradients = grad_fn(state.params)
    updates, new_opt_state = optimizer.update(gradients, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    new_state = TrainingState(
        params=new_params,
        opt_state=new_opt_state,
        rng_key=next_rng_key,
    )
    metrics = {
        'loss': loss,
        'accuracy': accuracy
    }

    return new_state, metrics


@partial(jax.jit, static_argnums=(3, 4))
def evaluate(
        state: TrainingState,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        model: hk.transform,
        classification: bool = False
) -> _Metrics:

    rng_key, _ = jax.random.split(state.rng_key, 2)

    if not classification:
        targets = inputs[:, :, 0]

    logits = model.apply(state.params, rng_key, inputs)
    loss = jnp.mean(cross_entropy_loss(logits, targets))
    accuracy = jnp.mean(compute_accuracy(logits, targets))

    metrics = {
        'loss': loss,
        'accuracy': accuracy
    }

    return metrics


def training_epoch(
        state: TrainingState,
        trainloader: DataLoader,
        model: hk.transform,
        optimizer: optax.GradientTransformation,
        classification: bool = False,
) -> Tuple[TrainingState, jnp.ndarray, jnp.ndarray]:

    batch_losses, batch_accuracies = [], []
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        inputs = jnp.array(inputs.numpy())
        targets = jnp.array(targets.numpy())
        state, metrics = update(
            state, inputs, targets,
            optimizer, model, classification
        )
        batch_losses.append(metrics['loss'])
        batch_accuracies.append(metrics['accuracy'])

    return (
        state,
        jnp.mean(jnp.array(batch_losses)),
        jnp.mean(jnp.array(batch_accuracies))
    )


def validation_epoch(
        state: TrainingState,
        testloader: DataLoader,
        model: hk.transform,
        classification: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    losses, accuracies = [], []
    for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
        inputs = jnp.array(inputs.numpy())
        targets = jnp.array(targets.numpy())
        metrics = evaluate(
            state, inputs, targets,
            model, classification
        )
        losses.append(metrics['loss'])
        accuracies.append(metrics['accuracy'])

    return jnp.mean(jnp.array(losses)), jnp.mean(jnp.array(accuracies))


def main():
    torch.random.manual_seed(SEED)
    key = jax.random.PRNGKey(SEED)
    rng, init_rng = jax.random.split(key)

    @hk.transform
    def forward(x) -> hk.transform:
        neural_net = S5Classifier(
            S5(
                STATE_SIZE,
                D_MODEL,
                N_BLOCKS,
                BASIS_MEASURE
            ),
            D_MODEL,
            ds.n_classes,
            N_LAYERS,
            DROPOUT_RATE,
            padded=False,
        )
        return hk.vmap(neural_net, split_rng=False)(x)

    ds = create_dataset(DATASET, BATCH_SIZE)
    optim = create_optimizer(ds)
    init_data = jnp.array(next(iter(ds.trainloader))[0].numpy())
    initial_params = forward.init(init_rng, init_data)
    initial_opt_state = optim.init(initial_params)

    state = TrainingState(
        params=initial_params,
        opt_state=initial_opt_state,
        rng_key=rng
    )

    for epoch in range(EPOCHS):
        print(f"[*] Training Epoch {epoch + 1}...")
        state, training_loss, training_accuracy = training_epoch(
            state,
            ds.trainloader,
            forward,
            optim,
            ds.classification
        )
        print(f"[*] Running Epoch {epoch + 1} Validation...")
        test_loss, test_accuracy = validation_epoch(
            state,
            ds.testloader,
            forward,
            ds.classification
        )
        print(f"\n=>> Epoch {epoch + 1} Metrics ===")
        print(
            f"\tTrain Loss: {training_loss:.5f} -- Train Accuracy:"
            f" {training_accuracy:.4f}\n\t Test Loss: {test_loss:.5f} --  Test"
            f" Accuracy: {test_accuracy:.4f}"
        )

if __name__ == '__main__':
  main()