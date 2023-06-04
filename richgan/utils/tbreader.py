from pathlib import Path
import pickle
import yaml
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
import tqdm
from .misc import flatten_dict_tree


class TBReader:
    default_size_guidance = {event_accumulator.TENSORS: 0}

    def __init__(self, path, size_guidance=None):
        self.path = path

        if size_guidance is None:
            size_guidance = self.default_size_guidance
        self.size_guidance = size_guidance

    def read_scalars(self):
        ea = event_accumulator.EventAccumulator(
            self.path, size_guidance=self.size_guidance
        )
        ea.Reload()

        result = {}
        for key in tqdm.tqdm(ea.Tags()["tensors"]):
            result[key] = np.array(
                [
                    (t.wall_time, t.step, tf.make_ndarray(t.tensor_proto))
                    for t in ea.Tensors(key)
                    if len(t.tensor_proto.tensor_shape.dim) == 0
                ]
            )
        return result


_check_types = [int, float, bool, str]


def _get_n_latest(array, n):
    ids = array[:, 1].argsort()
    return array[ids][-n:, 2]


def min_avg_ks(tb_data):
    return tb_data["ks_overall_avg/val"][:, 2].min()


def avg_ks_mean5latest(tb_data):
    return _get_n_latest(tb_data["ks_overall_avg/val"], 5).mean()


_default_aggregations = [min_avg_ks, avg_ks_mean5latest]


class BaseAggregator:
    def __init__(
        self,
        pattern,
        log_tag,
        logs_root,
        configs_root,
        config_fname,
    ):
        self.pattern = pattern
        self.log_tag = log_tag
        self.logs_root = Path(logs_root)
        self.configs_root = Path(configs_root)
        self.config_fname = config_fname

        self.log_paths = [
            p / self.log_tag for p in sorted(list(self.logs_root.glob(self.pattern)))
        ]
        self.config_files = sorted(
            list(self.configs_root.glob(f"{self.pattern}/{self.config_fname}"))
        )

        assert len(self.log_paths) == len(self.config_files)
        for lp, cp in zip(self.log_paths, self.config_files):
            assert (
                lp.parts[len(self.logs_root.parts)]
                == cp.parts[len(self.configs_root.parts)]
            )
        self._read_configs()

    @staticmethod
    def load_config(fname):
        with open(fname, "r") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)
        return config

    @staticmethod
    def _lists_to_dicts(tree):
        assert isinstance(tree, dict)

        types = set()
        for k in tree.keys():
            if isinstance(tree[k], dict):
                types = types.union(BaseAggregator._lists_to_dicts(tree[k]))
            elif isinstance(tree[k], list):
                tree[k] = {i: v for i, v in enumerate(tree[k])}
                types = types.union(BaseAggregator._lists_to_dicts(tree[k]))
            else:
                types.add(type(tree[k]))
        return types

    def _read_configs(self):
        self.configs = [
            BaseAggregator.load_config(f.as_posix()) for f in self.config_files
        ]

        types = set()
        for config in self.configs:
            types = types.union(BaseAggregator._lists_to_dicts(config))

        assert all(t in _check_types for t in types)

        flat_configs = [
            {
                ":".join(key.__repr__() for key in path): val
                for path, val in flatten_dict_tree(config)
            }
            for config in self.configs
        ]

        config_data = pd.concat(
            [pd.Series(flat_config) for flat_config in flat_configs], axis=1
        ).T

        def is_nontrivial(col):
            return len(col.unique()) > 1

        self.config_data = config_data.T.loc[config_data.apply(is_nontrivial, axis=0)].T


class TBAggregator(BaseAggregator):
    def __init__(
        self,
        pattern,
        log_tag="main",
        logs_root="logs",
        configs_root="saved_models",
        config_fname="config.yaml",
        aggregate_fns=None,
    ):
        super().__init__(
            pattern=pattern,
            log_tag=log_tag,
            logs_root=logs_root,
            configs_root=configs_root,
            config_fname=config_fname,
        )
        self.aggregate_fns = {f.__name__: f for f in _default_aggregations}
        if aggregate_fns is not None:
            self.aggregate_fns.update(aggregate_fns)

    @staticmethod
    def _merge_dicts(dicts):
        keys = set()
        for d in dicts:
            keys = keys.union(d.keys())

        result = {}
        for k in keys:
            result[k] = np.concatenate([d[k] for d in dicts if k in d], axis=0)
        return result

    def aggregate(self):
        aggregated_vals = []
        for log_path in self.log_paths:
            aggregated_vals.append(dict())
            tb_data = self._merge_dicts(
                [
                    TBReader(tbfile.as_posix()).read_scalars()
                    for tbfile in log_path.iterdir()
                    if tbfile.is_file()
                ]
            )
            for aggr_name, aggr_fn in self.aggregate_fns.items():
                try:
                    aggregated_vals[-1][aggr_name] = aggr_fn(tb_data)
                except KeyError as e:
                    print("Exception caught in TBAggregator:")
                    print(
                        f"{e.__class__.__name__}: {e}. Available keys:",
                        ", ".join(tb_data.keys()),
                    )
                    aggregated_vals[-1][aggr_name] = np.nan

        aggregated_vals = pd.concat([pd.Series(v) for v in aggregated_vals], axis=1).T

        return pd.concat([self.config_data, aggregated_vals], axis=1)


class RawEvalDataAggregator(BaseAggregator):
    def __init__(
        self,
        pattern,
        log_tag="eval",
        logs_root="logs",
        configs_root="saved_models",
        config_fname="config.yaml",
        epoch="latest",
        raw_eval_data_filename="raw_output_dict.pkl",
        summary_key="EfficiencyMaker:1",
    ):
        super().__init__(
            pattern=pattern,
            log_tag=log_tag,
            logs_root=logs_root,
            configs_root=configs_root,
            config_fname=config_fname,
        )
        self.epoch = epoch
        self.raw_eval_data_filename = raw_eval_data_filename
        self.summary_key = summary_key

        def find_epoch(path, epoch):
            candidates = [p for p in path.glob("*") if p.is_dir()]
            epochs = [int(cand.name) for cand in candidates]
            if epoch == "latest":
                i_epoch = np.argmax(epochs)
            elif isinstance(epoch, int):
                i_epoch = epochs.index(epoch)
            else:
                raise NotImplementedError(
                    f"RawEvalDataAggregator::find_epoch: Unsupported value ({epoch}) "
                    f"or type ({type(epoch)}) for the `epoch` parameter"
                )
            return candidates[i_epoch]

        self.raw_eval_data_files = [
            find_epoch(path / "pdf", self.epoch) / self.raw_eval_data_filename
            for path in self.log_paths
        ]
        for f in self.raw_eval_data_files:
            assert f.exists() and f.is_file()

        def load_pickle(fname):
            with open(fname, "rb") as f:
                data = pickle.load(f)
            return data

        self.raw_eval_data = [load_pickle(f) for f in self.raw_eval_data_files]
        assert len(self.raw_eval_data) == len(self.config_data)
        self.aggregated_data = []
        for data_entry, (_, config_entry) in zip(
            self.raw_eval_data, self.config_data.iterrows()
        ):
            for figure_name, data in data_entry[self.summary_key].items():
                self.aggregated_data.append(
                    pd.concat(
                        [
                            data.reset_index(),
                            pd.DataFrame(config_entry)
                            .T.iloc[np.zeros(len(data))]
                            .reset_index(drop=True),
                            pd.Series([figure_name] * len(data), name="figure_name"),
                        ],
                        axis=1,
                    )
                )
        self.aggregated_data = pd.concat(
            self.aggregated_data, axis=0, ignore_index=True
        )
