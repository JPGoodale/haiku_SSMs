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
from typing import Tuple, NamedTuple, MutableMapping, Any


_Metrics = MutableMapping[str, Any]

STATE_SIZE: int = 64
D_MODEL: int = 128
N_LAYERS: int = 4
EPOCHS: int = 100
BATCH_SIZE: int = 128
LEARNING_RATE: float = 0.001
WEIGHT_DECAY: float = 0.01


@partial(jnp.vectorize, signature="(c),()->()")
def cross_entropy_loss(logits, label):
    one_hot_label = jax.nn.one_hot(label, num_classes=logits.shape[0])
    return -jnp.sum(one_hot_label * logits)


@partial(jnp.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    return jnp.argmax(logits) == label


class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState
    rng_key: jnp.ndarray
    step: jnp.ndarray


def init_state(
        model: hk.transform,
        rng: jnp.ndarray,
        init_data: jnp.ndarray
) -> TrainingState:
    rng, init_rng = jax.random.split(rng)
    initial_params = model.init(init_rng, init_data)
    initial_opt_state = optimizer.init(initial_params)
    return TrainingState(
        params=initial_params,
        opt_state = initial_opt_state,
        rng_key=rng,
        step=jnp.array(0)
    )


optimizer = optax.adam(LEARNING_RATE)


def train_epoch(
        state: TrainingState,
        trainloader: DataLoader,
        model: hk.transform,
        classification: bool = False,
) -> Tuple[TrainingState, jnp.ndarray, jnp.ndarray]:
    batch_losses, batch_accuracies = [], []

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        inputs = jnp.array(inputs.numpy())
        targets = jnp.array(targets.numpy())
        state, metrics = train_step(
            state, inputs, targets,
            model, classification
        )
        batch_losses.append(metrics['loss'])
        batch_accuracies.append(metrics['accuracy'])

    return (
        state,
        jnp.mean(jnp.array(batch_losses)),
        jnp.mean(jnp.array(batch_accuracies))
    )


@partial(jax.jit, static_argnums=(3, 4))
def train_step(
        state: TrainingState,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
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
        step=state.step + 1
    )
    metrics = {
        'step': state.step,
        'loss': loss,
        'accuracy': accuracy
    }

    return new_state, metrics


def validate(
        state: TrainingState,
        testloader: DataLoader,
        model: hk.transform,
        classification: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    losses, accuracies = [], []

    for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
        inputs = jnp.array(inputs.numpy())
        targets = jnp.array(targets.numpy())
        metrics = eval_step(
            state, inputs, targets,
            model, classification
        )
        losses.append(metrics['loss'])
        accuracies.append(metrics['accuracy'])

    return jnp.mean(jnp.array(losses)), jnp.mean(jnp.array(accuracies))



@partial(jax.jit, static_argnums=(3, 4))
def eval_step(
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


def train(dataset: str, seed: int, dplr: bool, measure: str):
    print("[*] Setting Randomness...")
    torch.random.manual_seed(seed)  # For dataloader order
    key = jax.random.PRNGKey(seed)
    key, rng, train_rng = jax.random.split(key, num=3)

    classification = "classification" in dataset

    create_dataset_fn = Datasets[dataset]
    trainloader, testloader, n_classes, seq_length, d_input = create_dataset_fn(
        batch_size=BATCH_SIZE
    )
    init_data = jnp.array(next(iter(trainloader))[0].numpy())

    @hk.transform
    def model(x):
        neural_net = S4Stack(
            S4(STATE_SIZE, measure, seq_length, dplr),
            D_MODEL,
            N_LAYERS,
            n_classes,
            classification=classification
        )
        return hk.vmap(neural_net, split_rng=False)(x)

    state = init_state(model, rng, init_data)

    for epoch in range(EPOCHS):
        print(f"[*] Training Epoch {epoch + 1}...")
        state, train_loss, train_accuracy = train_epoch(
            state, trainloader, model, classification
        )
        print(f"[*] Running Epoch {epoch + 1} Validation...")
        test_loss, test_accuracy = validate(
            state, testloader, model, classification
        )

        print(f"\n=>> Epoch {epoch + 1} Metrics ===")
        print(
            f"\tTrain Loss: {train_loss:.5f} -- Train Accuracy:"
            f" {train_accuracy:.4f}\n\t Test Loss: {test_loss:.5f} --  Test"
            f" Accuracy: {test_accuracy:.4f}"
        )

train(dataset='kmnist', seed=0, dplr=False, measure='legs')


