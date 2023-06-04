from collections.abc import Iterable
import numpy as np
import pandas as pd
from .mpl_setup import plt

# import os
# print(os.getcwd())
#plt.style.use('matplotlib_cern.rc')
from matplotlib.ticker import FuncFormatter, FixedLocator
from .plots_base import PlotMakerBase


class EfficiencyMaker(PlotMakerBase):
    """
    A class for creating efficiency summary plots.

    This is intended to be configured at creation and then used by simply calling the make_figures
    public method, which only takes the data to make the plots with (+ an optional dictionary to
    record the raw data in). The make_figures method yields a generator of (name, figure) pairs.

    Public methods:
      - make_figures
    """

    def __init__(
        self,
        period_in_epochs,
        bins,
        figure_args,
        errorbar_common_args,
        errorbar_real_args,
        errorbar_fake_args,
        thresholds,
        make_ratio,
        name_prefix,
        bins_2d,
        per_bin_thresholds,
        **kwargs,
    ):
        """
        Constructor.

        Arguments:
        period_in_epochs -- for external use by the SummaryMetricsMaker, can be None
        bins -- integer, specifying the amount of intervals (bins) along the X-axis
        figure_args -- a dictionary of arguments passed to plt.figure()
        errorbar_common_args -- arguments passed to the plt.errorbar() calls
        errorbar_real_args -- plt.errorbar arguments that are specific for the calls on the real
            data component (only used when make_ratio=False)
        errorbar_fake_args -- plt.errorbar arguments that are specific for the calls on the fake
            data component (only used when make_ratio=False)
        thresholds -- a number or a list of numbers; quantiles at which the efficiencies will be
            calculated
        make_ratio -- boolean flag denoting whether individual efficiencies or their ratios should
            be plotted. When plotting individual efficiencies, a separate plot is created for each
            of the thresholds (which may result in a handful of images). When plotting ratios, the
            ratios for various thresholds are superimposed.
        name_prefix -- a string to prepend to the name when yielding a (name, figure) pair from the
            make_figures method
        bins_2d -- None or an integer specifying the amount of bins for the second feature to work
            in the 2d-bin regime. When set, all possible ordered pairs of features are considered.
            Given a pair, the first feature in the pair corresponds to the X-axis of the plot. The
            second feature is split into bins_2d bins and plotting is done separately for each of
            such bins.
        per_bin_thresholds -- boolean flag, specifying whether threshold quantiles are calculated
            for the entire X-axis or separately in each of the X-axis bins
        """
        super().__init__(**kwargs)
        self.period_in_epochs = period_in_epochs
        self.bins = bins
        self.figure_args = figure_args
        self.errorbar_common_args = errorbar_common_args
        self.errorbar_real_args = errorbar_real_args
        self.errorbar_fake_args = errorbar_fake_args
        self.thresholds = thresholds
        self.make_ratio = make_ratio
        self.name_prefix = name_prefix
        self.bins_2d = bins_2d
        self.per_bin_thresholds = per_bin_thresholds

        if make_ratio and (self.errorbar_real_args or self.errorbar_fake_args):
            print(
                "WARNING: *************************************************************************"
            )
            print(
                "WARNING: *** Real and fake args are ignored when making efficiency ratio plots ***"
            )
            print(
                "WARNING: *************************************************************************"
            )

    def _make_efficiency_figure(
        self,
        real_column,
        fake_column,
        feature_column,
        weight_column,
        quantiles,
        name_suffix,
        title_suffix,
        raw_output_dict,
        colors,
        markers,
        reverse
    ):
        bins = np.quantile(
            feature_column.to_numpy(), np.linspace(0.0, 1.0, self.bins + 1)
        )
        if reverse:
            end_str = "_reverse"
        else:
            end_str = ""

        name = f"{self.name_prefix}_{real_column.name}_vs_{feature_column.name}_at_{quantiles}"+end_str
        title = "{} efficiency{} vs {}{}".format(
            real_column.name,
            " ratio" if self.make_ratio else "",
            feature_column.name,
            f" at {quantiles}" if not isinstance(quantiles, Iterable) else "",
        )
        if name_suffix is not None:
            name = f"{name}_{name_suffix}"
        if title_suffix is not None:
            title = f"{title}; {title_suffix}"
        if not isinstance(quantiles, Iterable):
            quantiles = [quantiles]

        if not self.per_bin_thresholds:
            thresholds = np.quantile(real_column, quantiles)

        df = pd.DataFrame(
            {
                "real": real_column.values,
                "fake": fake_column.values,
                "feature": feature_column.values,
                "weight": weight_column.values,
            }
        )
        df["bin"] = pd.cut(df["feature"], bins=bins)
        group = df.groupby("bin")

        def calculate_efficiencies_or_their_ratios(df, reverse):
            total = df["weight"].sum()
            if total <= 0:
                result = pd.Series(
                    [np.nan] * (len(quantiles) * (3 if self.make_ratio else 6))
                )
                if self.make_ratio:
                    result.index = (
                        [f"eff_ratio_{q}" for q in quantiles]
                        + [f"eff_ratio_err_low_{q}" for q in quantiles]
                        + [f"eff_ratio_err_high_{q}" for q in quantiles]
                    )
                else:
                    result.index = (
                        sum(([f"eff_real_{q}", f"eff_fake_{q}"] for q in quantiles), [])
                        + sum(
                            (
                                [f"eff_real_err_low_{q}", f"eff_fake_err_low_{q}"]
                                for q in quantiles
                            ),
                            [],
                        )
                        + sum(
                            (
                                [f"eff_real_err_high_{q}", f"eff_fake_err_high_{q}"]
                                for q in quantiles
                            ),
                            [],
                        )
                    )
                return result

            if self.per_bin_thresholds:
                thresholds_ = np.quantile(df["real"], quantiles)
            else:
                thresholds_ = thresholds

            if reverse:
                passed = pd.concat(
                    [
                        ((df[["real", "fake"]] <= thr) * df[["weight"]].to_numpy()).sum(
                            axis=0
                        )
                        for thr in thresholds_
                    ],
                    axis=0,
                ).clip(lower=0.0)
            else:
                passed = pd.concat(
                    [
                        ((df[["real", "fake"]] >= thr) * df[["weight"]].to_numpy()).sum(
                            axis=0
                        )
                        for thr in thresholds_
                    ],
                    axis=0,
                ).clip(lower=0.0)
            if self.make_ratio:
                efficiencies = ((passed + 0.5) / (total + 1.0)).clip(0, 1)
            else:
                efficiencies = (passed / total).clip(0, 1)

            efficiencies.index = sum(
                ([f"eff_real_{q}", f"eff_fake_{q}"] for q in quantiles), []
            )

            if self.make_ratio:
                # Calculating ratio with 1-sigma confidence interval using the formula from here:
                # https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/binoraci.htm
                eff_real = efficiencies[[f"eff_real_{q}" for q in quantiles]].to_numpy()
                eff_fake = efficiencies[[f"eff_fake_{q}" for q in quantiles]].to_numpy()

                efficiency_ratios = eff_fake / eff_real

                exp_radical = np.exp(
                    (
                        (1 - eff_real) / (total * eff_real)
                        + (1 - eff_fake) / (total * eff_fake)
                    )
                    ** 0.5
                )
                errors_low = efficiency_ratios - efficiency_ratios / exp_radical
                errors_high = efficiency_ratios * exp_radical - efficiency_ratios



                efficiency_ratios = pd.Series(
                    efficiency_ratios, index=[f"eff_ratio_{q}" for q in quantiles]
                )


                errors_low = pd.Series(
                    errors_low, index=[f"eff_ratio_err_low_{q}" for q in quantiles]
                )
                errors_high = pd.Series(
                    errors_high, index=[f"eff_ratio_err_high_{q}" for q in quantiles]
                )


                result = pd.concat([efficiency_ratios, errors_low, errors_high], axis=0)

            else:
                # Calculating 1-sigma Wilson confidence interval as `mode +\- delta`
                mode = (efficiencies + 1.0 / (2.0 * total)) / (1.0 + 1.0 / total)
                delta = (
                    efficiencies * (1 - efficiencies) / total + 1.0 / (4.0 * total ** 2)
                ).clip(lower=0) ** 0.5 / (1.0 + 1.0 / total)

                errors_low = efficiencies - (mode - delta)
                errors_high = (mode + delta) - efficiencies

                errors_low.index = sum(
                    (
                        [f"eff_real_err_low_{q}", f"eff_fake_err_low_{q}"]
                        for q in quantiles
                    ),
                    [],
                )
                errors_high.index = sum(
                    (
                        [f"eff_real_err_high_{q}", f"eff_fake_err_high_{q}"]
                        for q in quantiles
                    ),
                    [],
                )

                result = pd.concat([efficiencies, errors_low, errors_high], axis=0)

            return result

        efficiencies = group.apply(calculate_efficiencies_or_their_ratios, reverse=reverse)
        if raw_output_dict is not None:
            assert name not in raw_output_dict
            raw_output_dict[name] = efficiencies.copy()

        figure = plt.figure(**self.figure_args)

        if self.make_ratio:
            plt.yscale("symlog", linthresh=1.0)
            #plt.grid(b=True, which="major", linewidth=1.25)
            # plt.grid(b=True, which="minor", linewidth=0.3)
            yaxis = plt.gca().yaxis
            yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x + 1}"))
            major_ticks = np.array(
                [-1.0, -0.5, 0.0, 0.5, 1.0, 9.0, 49.0, 99.0, 499.0, 999.0]
            )
            minor_ticks = np.concatenate(
                [
                    np.linspace(
                        l, r, 5 if i < 4 else (8 if i == 4 else 10), endpoint=False
                    )
                    for i, (l, r) in enumerate(zip(major_ticks[:-1], major_ticks[1:]))
                ],
                axis=0,
            )
            yaxis.set_major_locator(FixedLocator(major_ticks))
            yaxis.set_minor_locator(FixedLocator(minor_ticks))

        for q in quantiles:
            if self.make_ratio:
                args = self.errorbar_common_args.copy()
                args["label"] = f'{q * 100}% {args.get("label", "")}'

                # Mokhnenko
                """
                args['markeredgewidth'] = 5
                args['linewidth'] = 3
                args['capsize'] = 2
                """
                args['color'] = colors[q]
                args['marker'] = markers[q]

                # print('markeredgewidth:', args['markeredgewidth'])

                y_value = (efficiencies[f"eff_ratio_{q}"] - 1.0)
                yerr = efficiencies[
                        [f"eff_ratio_err_low_{q}", f"eff_ratio_err_high_{q}"]
                    ].T.to_numpy()
                plt.errorbar(
                    x=efficiencies.index.categories.mid,
                    y=y_value,
                    xerr=(
                        efficiencies.index.categories.right
                        - efficiencies.index.categories.left
                    )
                    / 2,
                    yerr=yerr,
                    **args,
                )
            else:
                real_args = dict(**self.errorbar_common_args, **self.errorbar_real_args)
                fake_args = dict(**self.errorbar_common_args, **self.errorbar_fake_args)
                for args in [real_args, fake_args]:
                    if len(quantiles) > 1:
                        args["label"] = f'{q * 100}% {args.get("label", "")}'

                plt.errorbar(
                    x=efficiencies.index.categories.mid,
                    y=efficiencies[f"eff_real_{q}"],
                    xerr=(
                        efficiencies.index.categories.right
                        - efficiencies.index.categories.left
                    )
                    / 2,
                    yerr=efficiencies[
                        [f"eff_real_err_low_{q}", f"eff_real_err_high_{q}"]
                    ].T.to_numpy(),
                    **real_args,
                )
                plt.errorbar(
                    x=efficiencies.index.categories.mid,
                    y=efficiencies[f"eff_fake_{q}"],
                    xerr=(
                        efficiencies.index.categories.right
                        - efficiencies.index.categories.left
                    )
                    / 2,
                    yerr=efficiencies[
                        [f"eff_fake_err_low_{q}", f"eff_fake_err_high_{q}"]
                    ].T.to_numpy(),
                    **fake_args,
                )
        if self.make_ratio:
            ymin, ymax = plt.gca().get_ylim()
            if ymin > -1.0:
                plt.ylim(bottom=-1.0)
            if ymax < 1.0:
                plt.ylim(top=1.0)


        if feature_column.name in ["Brunel_P", "nTracks_Brunel", "P_T"]:
            plt.xscale("log")
        # Mokhnenko
        ax = plt.gca()


        if reverse:
            # plt.ylim(bottom=-1.0, top=1.0)
            ax.set_ylim([-1, 1])


        """
        plt.text(0.25, 0.9, 'LHCb Simulation', fontsize=22, horizontalalignment='center',
               verticalalignment='center', transform=ax.transAxes)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.legend(fontsize=22, ncol=1)
        font_s = 25 # 22
        title = title.replace('Brunel_','')
        y_label = title.split(' vs ')[0]
        x_label = title.split(' vs ')[1]
        ax.set_xlabel(x_label, fontsize=font_s-2, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=font_s-2, fontweight='bold')
        # plt.title(title, fontsize=font_s-2, fontweight='bold')
        #Mokhnenko
        plt.tight_layout(pad=3.0, w_pad=3.0, h_pad=0.2)
        plt.xticks(fontsize=font_s-2)
        plt.yticks(fontsize=font_s-2)
        """
        plt.text(0.05, 0.9, r'LHCb Simulation''\n'r'Preliminary',  fontsize=30,
                 verticalalignment='center', transform=ax.transAxes)
        plt.axhline(y=0, color='black', linestyle='-')
        plt.legend(ncol=1, loc='upper right')
        title = title.replace('Brunel_', '')
        y_label = title.split(' vs ')[0]

        if reverse:
            y_label = y_label.replace('efficiency ratio', 'rev efficiency ratio')

        x_label = title.split(' vs ')[1].replace('ETA', '$\eta$')
        if x_label == 'P':
            x_label = 'P(MeV)'
        ax.set_xlabel(x_label, fontweight='bold', fontsize=32)
        ax.set_ylabel(y_label, fontweight='bold', fontsize=32)
        # ax.set_xlabel(y_label+' vs '+x_label, fontweight='bold')
        # plt.title(y_label, fontsize=30, fontweight='bold')
        # Mokhnenko
        plt.tight_layout(pad=3.0, w_pad=3.0, h_pad=0.2)
        plt.xticks()
        plt.yticks()


        return name, figure

    def make_figures(
        self, features, targets_real, targets_fake, weights, raw_output_dict=None
    ):
        if self.bins_2d is None:
            feature_columns_2d = [None]
        else:
            feature_columns_2d = features.columns

        for feature_column_2d in feature_columns_2d:
            if self.bins_2d is not None:
                bins_2d_edges = np.quantile(
                    features[feature_column_2d].to_numpy(),
                    np.linspace(0.0, 1.0, self.bins_2d + 1),
                )
                bins_2d = pd.cut(features[feature_column_2d], bins=bins_2d_edges)
                categories = list(bins_2d.dtype.categories)
                selections = [(bins_2d == cat).to_numpy() for cat in categories]
            else:
                categories = [None]
                selections = [np.ones(shape=(len(features),), dtype=bool)]

            for cat, sel in zip(categories, selections):
                if cat is None:
                    name_suffix = None
                    title_suffix = None
                else:
                    name_suffix = f"{feature_column_2d}_in_{cat.left}_{cat.right}"
                    title_suffix = f"{feature_column_2d} in ({cat.left}, {cat.right}]"


                for target_column in targets_real.columns:
                    for feature_column in features.columns:
                        if feature_column == feature_column_2d:
                            continue

                        quantiles_list = [[0.75, 0.9, 0.95], [0.05, 0.10, 0.25], [0.75, 0.9, 0.95], [0.05, 0.10, 0.25]]

                        colors_list  = [{0.75: 'g', 0.9: 'b', 0.95: 'r'}, {0.05: 'r', 0.10: 'b', 0.25: 'g'},
                                        {0.75: 'g', 0.9: 'b', 0.95: 'r'}, {0.05: 'r', 0.10: 'b', 0.25: 'g'}]
                        markers_list  = [{0.75: 'v', 0.9: 's', 0.95: 'o'}, {0.05: 'o', 0.10: 's', 0.25: 'v'},
                                         {0.75: 'v', 0.9: 's', 0.95: 'o'}, {0.05: 'o', 0.10: 's', 0.25: 'v'}]
                        reverse_list  = [False, False, True, True]

                        if self.make_ratio:
                            for num in range(0, len(quantiles_list)):
                                yield self._make_efficiency_figure(
                                    real_column=targets_real[target_column].loc[sel],
                                    fake_column=targets_fake[target_column].loc[sel],
                                    feature_column=features[feature_column].loc[sel],
                                    weight_column=weights.loc[sel],
                                    # quantiles=self.thresholds,
                                    quantiles=quantiles_list[num],
                                    name_suffix=name_suffix,
                                    title_suffix=title_suffix,
                                    raw_output_dict=raw_output_dict,
                                    colors = colors_list[num],
                                    markers = markers_list[num],
                                    reverse=reverse_list[num],
                            )

                        else:
                            # for threshold in self.thresholds:
                            for num in range(0, len(quantiles_list)):
                                for threshold in quantiles_list[num]:
                                    yield self._make_efficiency_figure(
                                        real_column=targets_real[target_column].loc[sel],
                                        fake_column=targets_fake[target_column].loc[sel],
                                        feature_column=features[feature_column].loc[sel],
                                        weight_column=weights.loc[sel],
                                        quantiles=threshold,
                                        name_suffix=name_suffix,
                                        title_suffix=title_suffix,
                                        raw_output_dict=raw_output_dict,
                                        colors=colors_list[num],
                                        markers=markers_list[num],
                                        reverse=reverse_list[num],
                                    )

