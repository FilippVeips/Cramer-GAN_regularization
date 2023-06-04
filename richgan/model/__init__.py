from .GAN_architectures import nn_factory
from .updaters import updater_factory
from .step_schedulers import step_scheduler_factory
from ..utils.factories import make_factory


class GANModel:
    def __init__(
        self,
        name,
        generator_config,
        discriminator_config,
        updater_config,
        step_scheduler_config,
    ):
        self.name = name
        self.generator = nn_factory(**generator_config)
        self.discriminator = nn_factory(**discriminator_config)

        self.updater = updater_factory(
            generator=self.generator, discriminator=self.discriminator, **updater_config
        )
        self.step_scheduler = step_scheduler_factory(
            discriminator_step=self.updater.disc_step,
            generator_step=self.updater.gen_step,
            **step_scheduler_config
        )
        self._callbacks = []

    def training_step(self, *argv, **kwargs):
        return self.step_scheduler.step(*argv, **kwargs)

    def save(self, prefix):
        self.updater.save_state(prefix)

    def load(self, prefix):
        self.updater.restore_state(prefix)

    @property
    def callbacks(self):
        callback_set = set(self._callbacks)
        for obj in [
            self.generator,
            self.discriminator,
            self.updater,
            self.step_scheduler,
        ]:
            if hasattr(obj, "callbacks"):
                callback_set.update(obj.callbacks)
        return list(callback_set)


gan_factory = make_factory([GANModel])


def create_gan(**kwargs):
    return gan_factory(classname="GANModel", **kwargs)
