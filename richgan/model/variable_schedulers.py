import numpy as np
from ..utils.factories import make_factory


class ExponentialDecayScheduler:
    def __init__(self, target_variable, decay_steps, decay_rate):
        self.target_variable = target_variable
        self.learning_rate = target_variable.numpy()
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def step(self, global_step):
        self.target_variable.assign(
            self.learning_rate * self.decay_rate ** (global_step / self.decay_steps)
        )


class ArctanSaturatingScheduler:
    def __init__(
        self,
        target_variable,
        magnitude,
        halfrise_steps,
        force_value_at_start,
        minimal_value,
    ):
        self.target_variable = target_variable
        self.magnitude = magnitude
        self.halfrise_steps = halfrise_steps
        self.minimal_value = minimal_value

        if force_value_at_start:
            self.step(0)

    def step(self, global_step):
        target_value = (
            self.magnitude / np.pi * 2 * np.arctan(global_step / self.halfrise_steps)
        )
        if target_value < self.minimal_value:
            target_value = self.minimal_value

        self.target_variable.assign(target_value)


variable_scheduler_factory = make_factory(
    [ExponentialDecayScheduler, ArctanSaturatingScheduler]
)
