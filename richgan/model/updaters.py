import tensorflow as tf
from .variable_schedulers import variable_scheduler_factory
from ..utils.factories import make_factory


class GANUpdaterBase:
    def __init__(
        self,
        generator,
        discriminator,
        gen_lr,
        disc_lr,
        gen_optimizer,
        disc_optimizer,
        gen_lr_scheduler,
        disc_lr_scheduler,
    ):
        self.generator = generator
        self.discriminator = discriminator

        self.gen_optimizer = (
            getattr(tf.optimizers, gen_optimizer)()
            if isinstance(gen_optimizer, str)
            else gen_optimizer
        )
        self.gen_optimizer.learning_rate.assign(gen_lr)

        self.disc_optimizer = (
            getattr(tf.optimizers, disc_optimizer)()
            if isinstance(disc_optimizer, str)
            else disc_optimizer
        )
        self.disc_optimizer.learning_rate.assign(disc_lr)

        self.callbacks = []

        if gen_lr_scheduler:
            self.gen_lr_scheduler = variable_scheduler_factory(
                target_variable=self.gen_optimizer.learning_rate, **gen_lr_scheduler
            )
            self.callbacks.append(self.gen_lr_scheduler.step)

        if disc_lr_scheduler:
            self.disc_lr_scheduler = variable_scheduler_factory(
                target_variable=self.disc_optimizer.learning_rate, **disc_lr_scheduler
            )
            self.callbacks.append(self.disc_lr_scheduler.step)

        self.objects_to_save = dict(
            generator=self.generator,
            discriminator=self.discriminator,
            gen_optimizer=self.gen_optimizer,
            disc_optimizer=self.disc_optimizer,
        )

    def write_summary(self, summary_writer, step):
        with summary_writer.as_default():
            tf.summary.scalar("gen_lr", self.gen_optimizer.learning_rate, step)
            tf.summary.scalar("disc_lr", self.disc_optimizer.learning_rate, step)

    def _get_checkpoint_obj(self):
        if not hasattr(self, "checkpoint"):
            self.checkpoint = tf.train.Checkpoint(**self.objects_to_save)
            del (
                self.objects_to_save
            )  # to ensure no other objects added to this dict afterwards
        return self.checkpoint

    def save_state(self, prefix):
        self._get_checkpoint_obj().write(prefix)

    def restore_state(self, prefix):
        self.restore_status = self._get_checkpoint_obj().read(prefix)
        self.callbacks.append(self._check_restore_status_callback)

    def _check_restore_status_callback(self, global_step):
        self.restore_status.assert_consumed()
        self.callbacks.remove(self._check_restore_status_callback)

    def get_losses(self, batch_main, batch_cond, batch_weights):
        raise NotImplementedError("Implement this method in a sub-class")

    @tf.function
    def gen_step(self, batch_main, batch_cond, batch_weights):
        with tf.GradientTape() as tape:
            losses = self.get_losses(batch_main, batch_cond, batch_weights)
        variables = self.generator.trainable_variables
        gradients = tape.gradient(losses["gen_loss"], variables)
        self.gen_optimizer.apply_gradients(zip(gradients, variables))

        return losses

    @tf.function
    def disc_step(self, batch_main, batch_cond, batch_weights):
        with tf.GradientTape() as tape:
            losses = self.get_losses(batch_main, batch_cond, batch_weights)
        variables = self.discriminator.trainable_variables
        gradients = tape.gradient(losses["disc_loss"], variables)
        self.disc_optimizer.apply_gradients(zip(gradients, variables))

        return losses


class GPUpdaterBase(GANUpdaterBase):
    def __init__(self, gp_lambda, gp_lambda_scheduler, **kwargs):
        super().__init__(**kwargs)

        self.gp_lambda = tf.Variable(gp_lambda, trainable=False)
        if gp_lambda_scheduler:
            self.gp_lambda_scheduler = variable_scheduler_factory(
                target_variable=self.gp_lambda, **gp_lambda_scheduler
            )
            self.callbacks.append(self.gp_lambda_scheduler.step)
        self.objects_to_save["gp_lambda"] = self.gp_lambda

    def write_summary(self, summary_writer, step):
        super().write_summary(summary_writer, step)
        with summary_writer.as_default():
            tf.summary.scalar("gp_lambda", self.gp_lambda, step)

    def gradient_penalty(
        self, function, batch_main, batch_fake, batch_cond, batch_weights
    ):
        batch_size = tf.shape(batch_main)[0]
        gp_alpha = tf.random.uniform(shape=(batch_size, 1), minval=0.0, maxval=1.0)
        interpolates = gp_alpha * batch_main + (1 - gp_alpha) * batch_fake
        with tf.GradientTape() as tape:
            tape.watch(interpolates)
            f_interpolates = function([interpolates, batch_cond])

        gradients = tf.reshape(
            tape.gradient(f_interpolates, interpolates), [batch_size, -1]
        )
        slopes = tf.norm(gradients, axis=1)
        return self.gp_lambda * tf.reduce_mean(tf.square(tf.maximum(slopes - 1, 0)))


class CramerUpdater(GPUpdaterBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @tf.function
    def cramer_critic(self, x, y):
        discriminated_x = self.discriminator(x)
        return tf.norm(discriminated_x - self.discriminator(y), axis=1) - tf.norm(
            discriminated_x, axis=1
        )

    @tf.function
    def get_losses(self, batch_main, batch_cond, batch_weights):
        fake_1 = self.generator(batch_cond)
        fake_2 = self.generator(batch_cond)

        gen_loss = tf.reduce_mean(
            (
                self.cramer_critic([batch_main, batch_cond], [fake_2, batch_cond])
                - self.cramer_critic([fake_1, batch_cond], [fake_2, batch_cond])
            )
            * batch_weights
        )

        gradient_penalty_term = self.gradient_penalty(
            function=lambda x: self.cramer_critic(x, [fake_2, batch_cond]),
            batch_main=batch_main,
            batch_fake=fake_1,
            batch_cond=batch_cond,
            batch_weights=batch_weights,
        )

        disc_loss = gradient_penalty_term - gen_loss

        return {"gen_loss": gen_loss, "disc_loss": disc_loss}


class WGANUpdater(GPUpdaterBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @tf.function
    def get_losses(self, batch_main, batch_cond, batch_weights):
        batch_fake = self.generator(batch_cond)

        gen_loss = tf.reduce_mean(
            (
                self.discriminator([batch_main, batch_cond])
                - self.discriminator([batch_fake, batch_cond])
            )
            * batch_weights
        )

        gradient_penalty_term = self.gradient_penalty(
            function=self.discriminator,
            batch_main=batch_main,
            batch_fake=batch_fake,
            batch_cond=batch_cond,
            batch_weights=batch_weights,
        )

        disc_loss = gradient_penalty_term - gen_loss

        return {"gen_loss": gen_loss, "disc_loss": disc_loss}


updater_factory = make_factory([CramerUpdater, WGANUpdater])
