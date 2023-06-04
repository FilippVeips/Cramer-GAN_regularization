import argparse
import yaml
import tensorflow as tf
from richgan.schemas import training_schema, evaluation_schema

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="richgan/configs/simple.config.yaml")
parser.add_argument("--gpu_num", type=str, default=None)
parser.add_argument("--schema", type=str, default="training")
parser.add_argument("--kwargs", type=str, default=None)
parser.add_argument("--oom_marker_file", type=str, default=None)
parser.add_argument("--no_uuid_suffix", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()

    kwargs = {}
    if args.kwargs is not None:
        with open(args.kwargs, "r") as f:
            kwargs.update(yaml.load(f, Loader=yaml.UnsafeLoader))

    try:
        if args.schema == "training":
            training_schema(
                gpu_num=args.gpu_num,
                config_file=args.config,
                uuid_suffix=not args.no_uuid_suffix,
                **kwargs,
            )
        elif args.schema == "evaluation":
            evaluation_schema(
                gpu_num=args.gpu_num,
                config_file=args.config,
                uuid_suffix=not args.no_uuid_suffix,
                **kwargs,
            )
        else:
            raise NotImplementedError(args.schema)
    except tf.errors.ResourceExhaustedError:
        if args.oom_marker_file is not None:
            with open(args.oom_marker_file, "w") as f:
                f.write("OOM\n")
        raise
