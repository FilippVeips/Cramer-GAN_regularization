from pathlib import Path
import shutil
import pytest

LAST_TEST_PATH = "last_test_outputs"


@pytest.fixture(scope="session")
def last_test_dir():
    the_path = Path(LAST_TEST_PATH)
    if the_path.exists():
        shutil.rmtree(the_path)
    the_path.mkdir()
    return the_path
