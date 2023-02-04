import dataclasses
import jax
import jax.numpy as jnp
import haiku as hk
from hippox.main import Hippo
from typing import Optional
from functions import init_log_steps, discretize_zoh, discretize_bilinear, apply_ssm, trunc_standard_normal


class S5(hk.Module):
    def __init__(self,
                 state_size: int,
                 n_features: int,
                 n_blocks: int,
                 basis_measure: str,
                 discretization: str = 'bilinear',
                 conj_sym: bool = True,
                 clip_eigns: bool = False,
                 bidirectional: bool = False,
                 step_rescale: float = 1.0,
                 name: Optional[str] = None
    ):
        super(S5, self).__init__(name=name)
        self.conj_sym = conj_sym
        self.bidirectional = bidirectional

        if conj_sym:
            _state_size = state_size // 2
        else:
            _state_size = state_size

        _hippo = Hippo(
            state_size=state_size,
            measure_family=basis_measure,
            conj_sym=conj_sym,
            block_diagonal=True,
            n_blocks=n_blocks,
        )
        _hippo_params = _hippo()

        self._lambda_real = hk.get_parameter(
            'lambda_real',
            [_state_size],
            init=_hippo.lambda_initializer('real')
        )
        self._lambda_imag = hk.get_parameter(
            'lambda_imaginary',
            [_state_size],
            init=_hippo.lambda_initializer('imaginary')
        )
        if clip_eigns:
            self._lambda = jnp.clip(self._lambda_real, None, -1e-4) + 1j * self._lambda_imag
        else:
            self._lambda = self._lambda_real + 1j * self._lambda_imag


        b_init = hk.initializers.VarianceScaling()
        b_shape = [state_size, n_features]
        b_init = b_init(b_shape, dtype=jnp.complex64)
        self._b = hk.get_parameter(
            'b',
            [_state_size, n_features, 2],
            init=_hippo.eigenvector_transform(b_init,  concatenate=True),
        )
        b = self._b[..., 0] + 1j * self._b[..., 1]

        c_init = hk.initializers.TruncatedNormal()
        c_shape = [n_features, state_size, 2]
        c_init = c_init(c_shape, dtype=jnp.complex64)
        self._c = hk.get_parameter(
            'c',
            [n_features, _state_size, 2],
            init=_hippo.eigenvector_transform(c_init, inverse=False, concatenate=True),
        )
        self._output_matrix = self._c[..., 0] + 1j * self._c[..., 1]

        self._d = hk.get_parameter(
            'd',
            [n_features,],
            init=hk.initializers.RandomNormal(stddev=1.0)
        )

        self._delta = hk.get_parameter(
            'delta',
            [_state_size, 1],
            init=init_log_steps
        )
        timescale = step_rescale * jnp.exp(self._delta[:, 0])

        if discretization == 'zoh':
            self._state_matrix, self._input_matrix = discretize_zoh(self._lambda, b, timescale)
        elif discretization == 'bilinear':
            self._state_matrix, self._input_matrix = discretize_bilinear(self._lambda, b, timescale)
        else:
            raise NotImplementedError('Discretization method {} not implemented'.format(discretization))

    def __call__(self, input_sequence):
        ys = apply_ssm(
            self._state_matrix,
            self._input_matrix,
            self._output_matrix,
            input_sequence,
            self.conj_sym,
            self.bidirectional
        )
        Du = jax.vmap(lambda u: self._d * u)(input_sequence)
        return ys + Du


@dataclasses.dataclass
class S5Block(hk.Module):
    ssm: S5
    d_model: int
    dropout_rate: float
    activation: str
    prenorm: bool
    istraining: bool = True
    name: Optional[str] = None

    def __post_init__(self):
        super(S5Block, self).__post_init__()
        self._norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self._linear1 = hk.Linear(self.d_model)
        self._linear2 = hk.Linear(self.d_model)

    def __call__(self, x):
        skip = x
        if self.prenorm:
            x = self._norm(x)

        x = self.ssm(x)
        if self.activation == 'full_glu':
            x = hk.dropout(hk.next_rng_key(), self.dropout_rate, jax.nn.gelu(x))
            x = self._linear1(x) * jax.nn.sigmoid(self._linear2(x))
            x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)
        elif self.activation == 'half_glu1':
            x = jax.nn.gelu(x)
            x = x * jax.nn.sigmoid(self._linear2(x))
            x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)
        elif self.activation == 'half_glu2':
            x1 = hk.dropout(hk.next_rng_key(), self.dropout_rate, jax.nn.gelu(x))
            x = x * jax.nn.sigmoid(self._linear2(x1))
            x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)
        elif self.activation == 'gelu':
            x = hk.dropout(hk.next_rng_key(), self.dropout_rate, jax.nn.gelu(x))
        else:
            raise NotImplementedError(
                "Activation: {} not implemented".format(self.activation))

        x = skip + x
        if not self.prenorm:
            x = self._norm(x)

        return x


@dataclasses.dataclass
class S5Stack(hk.Module):
    ssm: S5
    d_model: int
    n_layers: int
    dropout_rate: float
    activation: str
    prenorm: bool
    istraining: bool = True
    name: Optional[str] = None

    def __post_init__(self):
        super(S5Stack, self).__post_init__(name=self.name)
        self._encoder = hk.Linear(self.d_model)
        self._layers = [
            S5Block(
                ssm=self.ssm,
                d_model=self.d_model,
                dropout_rate=self.dropout_rate,
                activation=self.activation,
                istraining=self.istraining,
                prenorm=self.prenorm,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, x):
        x = self._encoder(x)
        for layer in self._layers:
            x = layer(x)
        return x


def masked_meanpool(x, lengths):
    L = x.shape[0]
    mask = jnp.arange(L) < lengths
    return jnp.sum(mask[..., None]*x, axis=0) / lengths

batch_masked_meanpool = jax.vmap(masked_meanpool)


@dataclasses.dataclass
class S5Classifier(hk.Module):
    ssm: S5
    d_model: int
    d_output: int
    n_layers: int
    dropout_rate: float
    padded: bool
    activation: str = 'half_glu2'
    mode: str = 'pool'
    prenorm: bool = True
    istraining: bool = True
    name: Optional[str] = None

    def __post_init__(self):
        super(S5Classifier, self).__post_init__(name=self.name)
        self._encoder = S5Stack(
            ssm=self.ssm,
            d_model=self.d_model,
            n_layers=self.n_layers,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            istraining=self.istraining,
            prenorm=self.prenorm,
        )
        self._decoder = hk.Linear(self.d_output)

    def __call__(self, x):
        if self.padded:
            x, length = x
        x = self._encoder(x)
        if self.mode == 'pool':
            if self.padded:
                x = masked_meanpool(x, length)
            else:
                x = jnp.mean(x, axis=0)
        elif self.mode == 'last':
            if self.padded:
                raise NotImplementedError("Mode must be in ['pool'] for self.padded=True (for now...)")
            else:
                x = x[-1]

        else:
            raise NotImplementedError("Mode must be in ['pool', 'last]")
        x = self._decoder(x)
        return jax.nn.log_softmax(x, axis=-1)
