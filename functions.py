import jax
import jax.numpy as jnp
import haiku as hk
from jax.numpy.linalg import inv, matrix_power
from functools import partial


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


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    def init(shape, dtype):
        uniform = hk.initializers.RandomUniform()
        return uniform(shape, dtype)*(jnp.log(dt_max) - jnp.log(dt_min)) + jnp.log(dt_min)
    return init


def layer_norm(x: jnp.ndarray) -> jnp.ndarray:
    ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    return ln(x)
