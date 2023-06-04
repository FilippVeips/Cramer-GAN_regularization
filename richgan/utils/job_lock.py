import os
from pathlib import Path


class JobLock:
    class JobRunningError(Exception):
        pass

    def __init__(self, job_id):
        self.job_id = str(job_id)
        self.lock_file = Path(f".joblock_{self.job_id}")

    def __enter__(self):
        if self.lock_file.exists():
            raise self.JobRunningError(
                "Lock file already exists - "
                f"some other process is already running a job with id '{self.job_id}'"
            )
        else:
            with self.lock_file.open("w") as f:
                f.write(f"Process {os.getpid()} running job with id '{self.job_id}'\n")
        return self.lock_file

    def __exit__(self, exc_type, exc_value, traceback):
        self.lock_file.unlink()
