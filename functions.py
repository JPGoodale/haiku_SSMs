import jax
import jax.numpy as jnp
import haiku as hk
from jax.numpy.linalg import inv, matrix_power
from functools import partial
from jax.numpy import broadcast_to
from jax.tree_util import tree_map
from typing import Optional

# Most of these are taken directly from either https://github.com/srush/annotated-s4 or
# https://github.com/lindermanlab/S5, with some minor changes here and there.

def add_batch(nest, batch_size: Optional[int]):
    broadcast = lambda x: broadcast_to(x, (batch_size,) + x.shape)
    return tree_map(broadcast, nest)


def layer_norm(x: jnp.ndarray) -> jnp.ndarray:
    ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    return ln(x)


def discretize_bilinear(Lambda, B_tilde, Delta):
    Identity = jnp.ones(Lambda.shape[0])
    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    Identity = jnp.ones(Lambda.shape[0])
    Lambda_bar = jnp.exp(Lambda * Delta)
    B_bar = (1/Lambda * (Lambda_bar-Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretize(A, B, step, mode="zoh"):
    if mode == "bilinear":
        num, denom = 1 + .5 * step*A, 1 - .5 * step*A
        return num / denom, step * B / denom
    elif mode == "zoh":
        return jnp.exp(step*A), (jnp.exp(step*A)-1)/A * B


def discrete_DPLR(Lambda, P, Q, B, C, step, L):
    # Convert parameters to matrices
    B = B[:, jnp.newaxis]
    Ct = C[jnp.newaxis, :]

    N = Lambda.shape[0]
    A = jnp.diag(Lambda) - P[:, jnp.newaxis] @ Q[:, jnp.newaxis].conj().T
    I = jnp.eye(N)

    # Forward Euler
    A0 = (2.0 / step) * I + A

    # Backward Euler
    D = jnp.diag(1.0 / ((2.0 / step) - Lambda))
    Qc = Q.conj().T.reshape(1, -1)
    P2 = P.reshape(-1, 1)
    A1 = D - (D @ P2 * (1.0 / (1 + (Qc @ D @ P2))) * Qc @ D)

    # A bar and B bar
    Ab = A1 @ A0
    Bb = 2 * A1 @ B

    # Recover Cbar from Ct
    Cb = Ct @ inv(I - matrix_power(Ab, L)).conj()
    return Ab, Bb, Cb.conj()


def s4d_ssm(A, B, C, step):
    N = A.shape[0]
    Abar, Bbar = discretize(A, B, step, mode="zoh")
    Abar = jnp.diag(Abar)
    Bbar = Bbar.reshape(N, 1)
    Cbar = C.reshape(1, N)
    return Abar, Bbar, Cbar


def scan_SSM(Ab, Bb, Cb, u, x0):
    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb @ u_k
        y_k = Cb @ x_k
        return x_k, y_k

    return jax.lax.scan(step, x0, u)


@partial(jax.jit, static_argnums=3)
def s4d_kernel_zoh(A, C, step, L):
    kernel_l = lambda l: (C * (jnp.exp(step * A) - 1) / A * jnp.exp(l * step * A)).sum()
    return jax.vmap(kernel_l)(jnp.arange(L)).real


@jax.jit
def cauchy(v, omega, lambd):
    """Cauchy matrix multiplication: (n), (l), (n) -> (l)"""
    cauchy_dot = lambda _omega: (v / (_omega - lambd)).sum()
    return jax.vmap(cauchy_dot)(omega)


def kernel_DPLR(Lambda, P, Q, B, C, step, L):
    # Evaluate at roots of unity
    # Generating function is (-)z-transform, so we evaluate at (-)root
    Omega_L = jnp.exp((-2j * jnp.pi) * (jnp.arange(L) / L))

    aterm = (C.conj(), Q.conj())
    bterm = (B, P)

    g = (2.0 / step) * ((1.0 - Omega_L) / (1.0 + Omega_L))
    c = 2.0 / (1.0 + Omega_L)

    # Reduction to core Cauchy kernel
    k00 = cauchy(aterm[0] * bterm[0], g, Lambda)
    k01 = cauchy(aterm[0] * bterm[1], g, Lambda)
    k10 = cauchy(aterm[1] * bterm[0], g, Lambda)
    k11 = cauchy(aterm[1] * bterm[1], g, Lambda)
    atRoots = c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)
    out = jnp.fft.ifft(atRoots, L).reshape(L)
    return out.real


def causal_convolution(u, K):
    assert K.shape[0] == u.shape[0]
    ud = jnp.fft.rfft(jnp.pad(u, (0, K.shape[0])))
    Kd = jnp.fft.rfft(jnp.pad(K, (0, u.shape[0])))
    out = ud * Kd
    return jnp.fft.irfft(out)[: u.shape[0]]


def linear_recurrence(A, B, C, inputs, prev_state):
    new_state = A @ prev_state + B @ inputs
    out = C @ new_state
    return out, new_state


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    def init(shape, dtype):
        uniform = hk.initializers.RandomUniform()
        return uniform(shape, dtype)*(jnp.log(dt_max) - jnp.log(dt_min)) + jnp.log(dt_min)
    return init


def init_log_steps(shape, dtype):
    H = shape[0]
    log_steps = []
    for i in range(H):
        log_step = log_step_initializer()(shape=(1,), dtype=dtype)
        log_steps.append(log_step)

    return jnp.array(log_steps)


def trunc_standard_normal(key, shape):
    H, P, _ = shape
    Cs = []
    for i in range(H):
        key, skey = jax.random.split(key)
        C = jax.nn.initializers.lecun_normal()(skey, shape=(1, P, 2))
        Cs.append(C)
    return jnp.array(Cs)[:, 0]


@jax.vmap
def binary_operator(q_i, q_j):
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def apply_ssm(A, B, C, input_sequence, conj_sym, bidirectional):
    A_elements = A * jnp.ones((input_sequence.shape[0], A.shape[0]))
    Bu_elements = jax.vmap(lambda u: B @ u)(input_sequence)

    _, xs = jax.lax.associative_scan(binary_operator, (A_elements, Bu_elements))

    if bidirectional:
        _, xs2 = jax.lax.associative_scan(binary_operator,
                                          (A_elements, Bu_elements),
                                          reverse=True)
        xs = jnp.concatenate((xs, xs2), axis=-1)

    if conj_sym:
        return jax.vmap(lambda x: 2*(C @ x).real)(xs)
    else:
        return jax.vmap(lambda x: (C @ x).real)(xs)
