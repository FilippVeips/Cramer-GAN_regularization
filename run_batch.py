import argparse
import yaml
from uuid import uuid4
import numpy as np
import time
import subprocess
from pathlib import Path
from richgan.utils.misc import flatten_dict_tree, restore_dict_tree

parser = argparse.ArgumentParser()
parser.add_argument("--base_config", type=str, required=True)
parser.add_argument("--gpu_num", type=str, default=None)
parser.add_argument("--schema", type=str, default="training")
parser.add_argument("--hpar_space_config", type=str, required=True)
parser.add_argument("--n_simultaneous_jobs", type=int, default=3)
parser.add_argument("--tmp_config", type=str, default="tmp.config.yaml")
parser.add_argument("--verbose", action="store_true", default=False)


def process_hpar_space_config(config):
    """
    Broadcast all leaf arrays to a common shape, then iterate through
    their values yielding a single config dictionary at each iteration
    """
    flat_hpar_config = flatten_dict_tree(config)
    broadcast_grid = np.broadcast(*(values for _, values in flat_hpar_config))

    for hpar_set in broadcast_grid:
        yield restore_dict_tree(
            zip(
                (paths for paths, _ in flat_hpar_config),
                (hpar.item() for hpar in hpar_set),
            )
        )


class LoggingProcess:
    def __init__(self, args, logfile):
        self.logfile = Path(logfile)
        self.logfile_stream = self.logfile.open("w")
        self.process = subprocess.Popen(
            args, stdout=self.logfile_stream, stderr=self.logfile_stream
        )
        self.returncode = None

    def check_finished(self):
        self.returncode = self.process.poll()
        if self.returncode is not None:
            self.logfile_stream.close()
            return True
        return False


def spawn_job(i_job, config, args, batch_path, oom_marker_file):
    tmp_config = batch_path / args.tmp_config
    tmp_config = tmp_config.with_name(
        f"{tmp_config.stem}_{i_job:06d}{tmp_config.suffix}"
    )

    if args.verbose:
        print("\n\tDumping config:")
        print("\t\t", config)
        print(f"\tinto {tmp_config.as_posix()}")
    with open(tmp_config.as_posix(), "w") as f:
        yaml.dump(config, stream=f)
    job_args = [
        "python",
        "run_single_job.py",
        "--config",
        args.base_config,
        "--schema",
        args.schema,
        "--kwargs",
        tmp_config.as_posix(),
        "--oom_marker_file",
        oom_marker_file.as_posix(),
    ]
    if args.gpu_num is not None:
        job_args += ["--gpu_num", args.gpu_num]

    logfile = batch_path / f"log_{i_job:06d}.stdout_stderr"
    if args.verbose:
        print(f"\nStarting job {i_job} with args:", " ".join(job_args))
        print(f"Logging to: {logfile.as_posix()}")

    return LoggingProcess(job_args, logfile)


if __name__ == "__main__":
    args = parser.parse_args()

    batch_id = str(uuid4())
    batch_path = Path(f"batch_{batch_id}")
    batch_path.mkdir()
    if args.verbose:
        print("***" * 15)
        print(f"***** Running batch with id {batch_id}")
    oom_marker_file = batch_path / ".oom"

    with open(args.hpar_space_config, "r") as f:
        hpar_space_config = yaml.load(f, Loader=yaml.UnsafeLoader)

    processes = []
    configs = {
        i_job: config
        for i_job, config in enumerate(process_hpar_space_config(hpar_space_config))
    }

    while True:
        finished_processes = [proc for proc in processes if proc.check_finished()]
        for proc in finished_processes:
            if args.verbose:
                print("\tProcess finished")
                print("\t\tWas streaming to:")
                print(f"\t\t\t{proc.logfile.as_posix()}")
                print(f"\t\tExit-code: {proc.returncode}")
            processes.remove(proc)

        if (len(processes) < args.n_simultaneous_jobs) and (len(configs) > 0):
            if oom_marker_file.exists():
                if args.verbose:
                    print(f"Found OOM marker file: {oom_marker_file.as_posix()}")
                    print(f"(n_simultaneous_jobs = {args.n_simultaneous_jobs})")
                assert args.n_simultaneous_jobs > 1
                args.n_simultaneous_jobs -= 1
                oom_marker_file.unlink()
                if args.verbose:
                    print(
                        f"(Setting n_simultaneous_jobs to {args.n_simultaneous_jobs})"
                    )
            else:
                i_job, config = configs.popitem()
                processes.append(
                    spawn_job(
                        i_job=i_job,
                        config=config,
                        args=args,
                        batch_path=batch_path,
                        oom_marker_file=oom_marker_file,
                    )
                )

        if (len(configs) == 0) and (len(processes) == 0):
            break
        else:
            time.sleep(1)

    if args.verbose:
        print("All done...")
