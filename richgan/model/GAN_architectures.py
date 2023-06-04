import tensorflow as tf
from .nn_architectures import build_mlp_architecture_with_two_inputs
from ..utils.factories import make_factory


class NNBase:
    @property
    def architecture(self):
        raise NotImplementedError("Define the architecture in a sub-class")


class GeneratorBase(tf.keras.layers.Layer, NNBase):
    def __init__(self, n_latent_dims, distribution, **kwargs):
        super().__init__(**kwargs)
        self.n_latent_dims = n_latent_dims
        self.distribution = distribution

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=True):
        if self.distribution == "normal":
            seed = tf.random.normal(shape=[tf.shape(inputs)[0], self.n_latent_dims])
        else:
            raise NotImplementedError(self.distribution)

        return self.architecture([seed, inputs], training=training)


class DiscriminatorBase(tf.keras.layers.Layer, NNBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @tf.function
    def call(self, inputs, training=True):
        return self.architecture(inputs, training=training)


class SimpleGenerator(GeneratorBase):
    def __init__(
        self,
        n_latent_dims,
        distribution,
        depth,
        width,
        activation,
        input_size,
        output_size,
        arch_name,
        **kwargs
    ):
        super().__init__(n_latent_dims, distribution, **kwargs)
        self._architecture = build_mlp_architecture_with_two_inputs(
            input_1_size=n_latent_dims,
            input_2_size=input_size,
            width=width,
            depth=depth,
            activation=activation,
            output_size=output_size,
            name=arch_name,
        )

    @property
    def architecture(self):
        return self._architecture


class SimpleDiscriminator(DiscriminatorBase):
    def __init__(
        self,
        depth,
        width,
        activation,
        input_size_main,
        input_size_cond,
        output_size,
        arch_name,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._architecture = build_mlp_architecture_with_two_inputs(
            input_1_size=input_size_main,
            input_2_size=input_size_cond,
            width=width,
            depth=depth,
            activation=activation,
            output_size=output_size,
            name=arch_name,
        )

    @property
    def architecture(self):
        return self._architecture


nn_factory = make_factory([SimpleDiscriminator, SimpleGenerator])
