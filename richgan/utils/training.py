from pathlib import Path
import re
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from .factories import make_factory
from .event_schedulers import global_check_step


class TrainingManager:
    def __init__(
        self,
        model,
        data_manager,
        batch_size,
        epochs,
        log_path,
        save_base_path,
        save_interval_in_epochs,
    ):
        self.model = model
        self.data_manager = data_manager
        self.batch_size = batch_size
        self.epochs = epochs
        self.log_path = log_path
        self.save_base_path = save_base_path
        self.save_interval_in_epochs = save_interval_in_epochs

        self.global_step = 0

        self._callbacks = []

        if self.log_path is not None:
            log_path = Path(self.log_path)
            self.summary_writer = tf.summary.create_file_writer(
                (log_path / self.model.name / "main").as_posix()
            )

        if self.save_base_path is not None:
            self.save_path = Path(self.save_base_path) / self.model.name
            if self.save_path.exists():
                if not self.load_checkpoint():
                    print(
                        "WARNING: Found no checkpoints in the existing folder",
                        self.save_path.as_posix(),
                    )

    def register_callback(self, callback):
        self._callbacks.append(callback)

    def save_model(self, global_step):
        save_every = 1
        if hasattr(self, "save_interval_in_epochs"):
            save_every = self.save_interval_in_epochs

        if global_check_step(global_step, save_every):
            self.model.save((self.save_path / f"epoch-{global_step:06d}").as_posix())

    def load_checkpoint(self, num=-1):
        cps = sorted(self.save_path.glob("epoch*.index"))
        if len(cps) == 0:
            return False
        cp = cps[num]
        cp = self.save_path / cp.stem

        (self.global_step,) = re.findall(r"\d+", cp.stem)
        self.global_step = int(self.global_step) + 1

        print("Loading cp:", cp.as_posix())
        self.model.load(cp.as_posix())
        return True

    @property
    def callbacks(self):
        callbacks_set = set(self._callbacks)

        for obj in [self.model, self.data_manager]:
            if hasattr(obj, "callbacks"):
                callbacks_set.update(obj.callbacks)

        return list(callbacks_set)

    def run_training_loop(self):
        while self.global_step < self.epochs:
            print(f"Starting epoch {self.global_step}", flush=True)

            batch_generator = self.data_manager.get_batch_generator(self.batch_size, drop_P_T=True)

            est_batch_number = int(
                np.ceil(len(self.data_manager.data_train) / self.batch_size)
            )

            epoch_losses = {}
            # TODO: find a way to make this with tf datasets
            for data_batch in tqdm(batch_generator, total=est_batch_number):
                losses = self.model.training_step(
                    tf.convert_to_tensor(
                        data_batch[self.data_manager.target_columns], dtype="float32"
                    ),
                    tf.convert_to_tensor(
                        data_batch[self.data_manager.feature_columns], dtype="float32"
                    ),
                    tf.convert_to_tensor(
                        data_batch[self.data_manager.weight_column], dtype="float32"
                    ),
                )
                for k, v in losses.items():
                    if k in epoch_losses:
                        epoch_losses[k] += v.numpy() * len(data_batch)
                    else:
                        epoch_losses[k] = v.numpy() * len(data_batch)

            for k in epoch_losses:
                epoch_losses[k] /= len(self.data_manager.data_train)
                print(f"\t{k}: {epoch_losses[k]}")
            if hasattr(self, "summary_writer"):
                with self.summary_writer.as_default():
                    for k, v in epoch_losses.items():
                        tf.summary.scalar(k, v, self.global_step)
                self.model.updater.write_summary(self.summary_writer, self.global_step)

            for callback in self.callbacks:
                callback(self.global_step)

            if hasattr(self, "save_base_path"):
                self.save_model(self.global_step)

            print(f"Finished epoch {self.global_step}")
            self.global_step += 1


tm_factory = make_factory([TrainingManager])


def create_training_manager(**kwargs):
    return tm_factory(classname="TrainingManager", **kwargs)
