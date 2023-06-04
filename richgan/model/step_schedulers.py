from ..utils.factories import make_factory


class KStepScheduler:
    def __init__(self, k, discriminator_step, generator_step):
        self.k = k
        self.discriminator_step = discriminator_step
        self.generator_step = generator_step
        self.step_counter = 0

    def step(self, *argv, **kwargs):
        if self.step_counter < self.k:
            self.step_counter += 1
            return self.discriminator_step(*argv, **kwargs)
        else:
            self.step_counter = 0
            return self.generator_step(*argv, **kwargs)


step_scheduler_factory = make_factory([KStepScheduler])
