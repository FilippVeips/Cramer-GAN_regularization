import numpy as np
import pandas as pd
from ..utils.factories import make_factory
from .kolmogorov_smirnov import ks_2samp_w


class ScalarMakerBase:
    def __init__(self):
        self.period_in_epochs = None

    def make_scalars(self, features, targets_real, targets_fake, weights):
        raise NotImplementedError("Re-implement this method in a sub-class.")


class WeightedKSMaker(ScalarMakerBase):
    def __init__(self, bins, period_in_epochs, **kwargs):
        super().__init__(**kwargs)
        self.bins = bins
        self.period_in_epochs = period_in_epochs

    def make_scalars(self, features, targets_real, targets_fake, weights):
        results_avg = pd.DataFrame()
        results_max = pd.DataFrame()
        for feature_column in features.columns:
            bins = pd.qcut(features[feature_column], self.bins)
            for target_column in targets_real.columns:
                df = pd.DataFrame(
                    {
                        "real": targets_real[target_column],
                        "fake": targets_fake[target_column],
                        "weight": weights,
                        "bin": bins,
                    }
                )
                group = df.groupby("bin")

                def calculate_ks(df):
                    try:
                        return ks_2samp_w(
                            df["real"].to_numpy(),
                            df["fake"].to_numpy(),
                            df["weight"].to_numpy(),
                            df["weight"].to_numpy(),
                        )
                    except AssertionError as e:
                        print("Assertion triggered:", e)
                        return np.nan

                ks_values = group.apply(calculate_ks)
                sizes = group["weight"].sum()

                selection = ~ks_values.isna()
                ks_values = ks_values[selection]
                sizes = sizes[selection]

                avg_ks = (ks_values * sizes).sum() / sizes.sum()
                max_ks = ks_values.max()

                results_avg.loc[(target_column, feature_column)] = avg_ks
                results_max.loc[(target_column, feature_column)] = max_ks

                name_base = f"ks_{target_column}_vs_{feature_column}_"
                yield name_base + "avg", avg_ks
                yield name_base + "max", max_ks

        assert not results_avg.isna().any().any()
        assert not results_max.isna().any().any()

        for axis in [0, 1]:
            for column, value in results_avg.mean(axis=axis).items():
                yield f"ks_aggregated_{column}_avg", value
            for column, value in results_max.max(axis=axis).items():
                yield f"ks_aggregated_{column}_max", value

        yield "ks_overall_avg", results_avg.mean().mean()
        yield "ks_overall_max", results_max.max().max()


scalar_maker_factory = make_factory([WeightedKSMaker])
