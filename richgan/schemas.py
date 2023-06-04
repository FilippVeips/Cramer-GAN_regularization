import yaml
import tensorflow as tf
from pathlib import Path
import uuid
from copy import deepcopy
from .configs.defaults import class_constructor_defaults
from .utils.data import create_data_manager
from .model import create_gan
from .utils.training import create_training_manager
from .utils.cuda_gpu_config import setup_gpu
from .utils.event_schedulers import global_periodic_scheduler
from .utils.misc import merge_dicts, flatten_dict_tree
from .metrics import create_summary_maker
from .utils.job_lock import JobLock


def _create_config_and_update_from_file(config_file, **kwargs):
    config = {}
    if config_file is not None:
        with open(config_file, "r") as f:
            config.update(yaml.load(f, Loader=yaml.UnsafeLoader))
    merge_dicts(dest=config, src=kwargs, overwrite=True)
    return config


def _id_from_config(config):
    name = config["create_gan"]["name"]

    def _hard_flatten(config, root=True):
        assert isinstance(config, dict) or isinstance(config, list)
        if root:
            assert isinstance(config, dict)
            config = deepcopy(config)

        if isinstance(config, dict):
            flat_tree = sorted(flatten_dict_tree(config))
            config = (value for _, value in flat_tree)

        for value in config:
            if isinstance(value, list):
                for i in range(len(value)):
                    if isinstance(value[i], list) or isinstance(value[i], dict):
                        value[i] = _hard_flatten(value[i], root=False)

        return flat_tree

    job_id = str(uuid.uuid5(uuid.UUID(int=0), str(_hard_flatten(config))))
    return f"{name}-{job_id}"


def training_schema(config_file=None, gpu_num=None, uuid_suffix=True, **kwargs):
    config = _create_config_and_update_from_file(config_file, **kwargs)
    job_id = _id_from_config(config) if uuid_suffix else config["create_gan"]["name"]
    try:
        with JobLock(job_id):
            setup_gpu(gpu_num)
            gan_config = deepcopy(config["create_gan"])
            if uuid_suffix:
                gan_config["name"] = job_id

            dm = create_data_manager(**config["create_data_manager"])
            gan = create_gan(**gan_config)
            tm = create_training_manager(
                model=gan, data_manager=dm, **config["create_training_manager"]
            )
            if hasattr(tm, "save_path"):
                config_save_path = tm.save_path / "config.yaml"
                defaults_save_path = tm.save_path / "defaults.yaml"
                if tm.save_path.exists():
                    assert config_save_path.exists()
                    assert defaults_save_path.exists()

                    with open(config_save_path, "r") as f:
                        restored_config = yaml.load(f, Loader=yaml.UnsafeLoader)
                    with open(defaults_save_path, "r") as f:
                        restored_defaults = yaml.load(f, Loader=yaml.UnsafeLoader)

                    assert restored_config == config
                    assert restored_defaults == class_constructor_defaults

                else:
                    tm.save_path.mkdir(parents=True)
                    with open(config_save_path, "w") as f:
                        yaml.dump(config, stream=f)
                    with open(defaults_save_path, "w") as f:
                        yaml.dump(class_constructor_defaults, stream=f)

            for summary_kwargs in config["create_summary_makers"]:
                tm.register_callback(
                    create_summary_maker(
                        model=gan,
                        data_manager=dm,
                        summary_writer=tm.summary_writer,
                        **summary_kwargs,
                    ).summary_callback
                )

            tm.run_training_loop()
    except JobLock.JobRunningError as e:
        print(e)


def evaluation_schema(config_file=None, gpu_num=None, uuid_suffix=True, **kwargs):
    config = _create_config_and_update_from_file(config_file, **kwargs)
    job_id = _id_from_config(config) if uuid_suffix else config["create_gan"]["name"]
    try:
        with JobLock(job_id):
            setup_gpu(gpu_num)
            gan_config = deepcopy(config["create_gan"])
            if uuid_suffix:
                gan_config["name"] = job_id
            tag = config.get("tag", "eval")

            dm = create_data_manager(**config["create_data_manager"])
            gan = create_gan(**gan_config)
            tm = create_training_manager(
                model=gan, data_manager=dm, **config["create_training_manager"]
            )
            assert tm.log_path is not None
            log_path = Path(tm.log_path) / tm.model.name / tag
            summary_writer = tf.summary.create_file_writer(log_path.as_posix())

            callbacks = [
                create_summary_maker(
                    model=gan,
                    data_manager=dm,
                    summary_writer=summary_writer,
                    figures_log_path=(log_path / "pdf").as_posix(),
                    **summary_kwargs,
                ).summary_callback
                for summary_kwargs in config["create_summary_makers"]
            ]

            global_periodic_scheduler.set_passthrough()
            for callback in callbacks:
                callback(tm.epochs - 1)
    except JobLock.JobRunningError as e:
        print(e)
