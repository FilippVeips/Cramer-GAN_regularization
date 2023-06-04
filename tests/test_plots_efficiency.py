import pytest
import numpy as np
import pandas as pd
from richgan.metrics.plots import plot_maker_factory
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


DS_SIZE = 1000
FEATURES = ["f1", "f2", "f3"]
TARGETS = ["t1", "t2", "t3", "t4", "t5"]
WEIGHT = "weight"
MAX_PLOTS_PER_TEST = 20


def generate_random_dataset(N, columns, generator, **kwargs):
    data = generator(size=(N, len(columns)), **kwargs)
    result = pd.DataFrame(data, columns=columns)
    if len(columns) == 1:
        result = result[columns[0]]
    return result


@pytest.fixture
def em_make_ratio(request):
    make_ratio = request.param
    assert isinstance(make_ratio, bool)
    return make_ratio


@pytest.fixture
def em_bins_2d(request):
    bins_2d = request.param
    assert bins_2d is None or isinstance(bins_2d, int)
    return bins_2d


@pytest.fixture
def em_per_bin_thresholds(request):
    per_bin_thresholds = request.param
    assert isinstance(per_bin_thresholds, bool)
    return per_bin_thresholds


@pytest.fixture
def efficiency_maker(em_make_ratio, em_bins_2d, em_per_bin_thresholds):
    return plot_maker_factory(
        classname="EfficiencyMaker",
        make_ratio=em_make_ratio,
        bins=3,
        thresholds=[0.33, 0.66],
        bins_2d=em_bins_2d,
        per_bin_thresholds=em_per_bin_thresholds,
    )


@pytest.fixture(scope="session")
def features():
    return generate_random_dataset(DS_SIZE, FEATURES, generator=np.random.uniform)


@pytest.fixture(scope="session")
def targets_real():
    return generate_random_dataset(DS_SIZE, TARGETS, generator=np.random.normal)


@pytest.fixture(scope="session")
def targets_fake():
    return generate_random_dataset(DS_SIZE, TARGETS, generator=np.random.normal)


@pytest.fixture
def weights():
    return generate_random_dataset(
        DS_SIZE, [WEIGHT], generator=np.random.uniform, low=0.8, high=1.2
    )


@pytest.mark.parametrize("em_make_ratio", [True, False], indirect=True)
@pytest.mark.parametrize("em_bins_2d", [None, 2], indirect=True)
@pytest.mark.parametrize("em_per_bin_thresholds", [True, False], indirect=True)
def test_efficiency_maker(
    efficiency_maker, features, targets_real, targets_fake, weights, last_test_dir
):
    save_path = (
        last_test_dir
        / f"{efficiency_maker.__class__.__name__}-{hex(hash(efficiency_maker))}"
    )
    save_path.mkdir()
    for i_plot, (name, figure) in enumerate(
        efficiency_maker.make_figures(
            features=features,
            targets_real=targets_real,
            targets_fake=targets_fake,
            weights=weights,
        )
    ):
        assert isinstance(name, str)
        assert isinstance(figure, Figure)
        figure.savefig((save_path / f"{name}.pdf").as_posix())
        plt.close(figure)
        if i_plot == MAX_PLOTS_PER_TEST - 1:
            break
