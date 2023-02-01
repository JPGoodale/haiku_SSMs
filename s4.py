import jax
import jax.numpy as jnp
import haiku as hk
import dataclasses
from hippox.main import Hippo
from typing import Optional
from functions import log_step_initializer,\
    kernel_DPLR, s4d_kernel_zoh, discrete_DPLR, \
    s4d_ssm, scan_SSM, causal_convolution, layer_norm, \
    linear_recurrence, add_batch


class S4(hk.Module):
    def __init__(self,
                 state_size: int,
                 measure: str,
                 seq_length: int,
                 dplr: bool,
                 inference_mode: bool = False,
                 name: Optional[str] = None
    ):
        super(S4, self).__init__(name=name)
        self._state_size = state_size
        self._inference_mode = inference_mode

        _hippo = Hippo(
            state_size=state_size,
            measure_family=measure,
            dplr=dplr,
        )
        _hippo_params = _hippo()
        self._lambda_real = hk.get_parameter(
            'lambda_real',
            [self._state_size,],
            init=_hippo.lambda_initializer('real')
        )
        self._lambda_imag = hk.get_parameter(
            'lambda_imaginary',
            [self._state_size,],
            init=_hippo.lambda_initializer('imaginary')
        )
        self._lambda = jnp.clip(self._lambda_real, None, -1e-4) + 1j * self._lambda_imag

        if dplr:
            self._p = hk.get_parameter(
                'p',
                [self._state_size],
                init=_hippo.low_rank_initializer()
            )

        self._b = hk.get_parameter(
            'b',
            [self._state_size],
            init=_hippo.b_initializer()
        )

        self._c = hk.get_parameter(
            'c',
            [self._state_size, 2],
            init=hk.initializers.RandomNormal(stddev=0.5**0.5)
        )
        self._c = self._c[..., 0] + 1j * self._c[..., 1]

        self._d = hk.get_parameter(
            'd',
            [1,],
            init=jnp.ones,
        )

        self._delta = hk.get_parameter(
            'delta',
            [1,],
            dtype=jnp.float32,
            init=log_step_initializer()
        )
        self._timescale = jnp.exp(self._delta)

        if not self._inference_mode:
            if dplr:
                self._kernel = kernel_DPLR(self._lambda, self._p, self._p, self._b, self._c, self._delta, seq_length)
            else:
                self._kernel = s4d_kernel_zoh(self._lambda, self._c, self._timescale, seq_length)
        else:
            if dplr:
                self._ssm = discrete_DPLR(self._lambda, self._p, self._p, self._b, self._c, self._delta, seq_length)
            else:
                self._ssm = s4d_ssm(self._lambda, self._b, self._c, self._timescale)

            self._state = hk.get_state('state', [self._state_size], jnp.complex64, jnp.zeros)


    def __call__(self, u):
        if not self._inference_mode:
            return causal_convolution(u, self._kernel) + self._d * u
        else:
            x_k, y_s = scan_SSM(*self._ssm, u[:, jnp.newaxis], self._state)
            hk.set_state('state', x_k)
            return y_s.reshape(-1).real + self._d * u



@dataclasses.dataclass
class S4Block(hk.Module):
    ssm: S4
    d_model: int
    dropout_rate: float
    prenorm: bool = True
    glu: bool = True
    istraining: bool = False
    inference_mode: bool = False
    name: Optional[str] = None

    def __call__(self, x):
        skip = x
        if self.prenorm:
            x = layer_norm(x)
        x = hk.vmap(self.ssm, in_axes=1, out_axes=1, split_rng=True)(x)
        x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)
        if self.glu:
            x = hk.Linear(self.d_model)(x) * jax.nn.sigmoid(hk.Linear(self.d_model)(x))
        else:
            x = hk.Linear(self.d_model)(x)
        x = skip + hk.dropout(hk.next_rng_key(), self.dropout_rate, x)
        if not self.prenorm:
            x = layer_norm(x)
        return x


@dataclasses.dataclass
class Embedding(hk.Module):
    n_embeddings: int
    n_features: int

    def __call__(self, x):
        y = hk.Embed(self.n_embeddings, self.n_features)(x[..., 0])
        return jnp.where(x > 0, y, 0.0)


@dataclasses.dataclass
class S4Stack(hk.Module):
    ssm: S4
    d_model: int
    n_layers: int
    d_output: int
    prenorm: bool = True
    dropout_rate: float = 0.0
    embedding: bool = False
    classification: bool = False
    istraining: bool = True
    inference_mode: bool = False
    name: Optional[str] = None

    def __post_init__(self):
        super(S4Stack, self).__post_init__(name=self.name)
        if self.embedding:
            self._encoder = Embedding(self.d_output, self.d_model)
        else:
            self._encoder = hk.Linear(self.d_model)
        self._decoder = hk.Linear(self.d_output)
        self._layers = [
            S4Block(
                ssm=self.ssm,
                prenorm=self.prenorm,
                d_model=self.d_model,
                dropout_rate=self.dropout_rate,
                istraining=self.istraining,
                inference_mode=self.inference_mode,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, x):
        if not self.classification:
            if not self.embedding:
                x = x / 255.0
            if not self.inference_mode:
                x = jnp.pad(x[:-1], [(1, 0), (0, 0)])
        x = self._encoder(x)
        for layer in self._layers:
            x = layer(x)
        if self.classification:
            x = jnp.mean(x, axis=0)
        x = self._decoder(x)
        return jax.nn.log_softmax(x, axis=-1)
