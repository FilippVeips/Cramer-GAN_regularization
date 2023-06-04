import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("Agg")
files_path = './richgan/metrics/plots'
# plt.style.use(os.path.join(files_path, 'matplotlibrc'))
from matplotlib.ticker import FuncFormatter, FixedLocator
import io
import PIL
from collections.abc import Iterable
import re

# абстрактный класс для наследования
class PlotMakerBase:
    def __init__(self):
        self.period_in_epochs = None

    def make_figures(
            self, features, targets_real, targets_fake, weights, raw_output_dict=None
    ):
        raise NotImplementedError("Re-implement this method in a sub-class.")


# здесь будем вводить для двух
class EfficiencyMakerDouble(PlotMakerBase):
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

    def _make_efficiency_figure(
            self,
            real_column,
            fake_column,
            fake_column2,
            feature_column,
            weight_column,
            quantiles,
            name_suffix,
            title_suffix,
            raw_output_dict,
            colors,
            markers,
            reverse,
            model_names = [],
    ):
        bins = np.quantile(
            feature_column.to_numpy(), np.linspace(0.0, 1.0, self.bins + 1)
        )
        if reverse:
            end_str = "_reverse"
        else:
            end_str = ""

        name = f"{self.name_prefix}_{real_column.name}_vs_{feature_column.name}_at_{quantiles}" + end_str
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

        df2 = pd.DataFrame(
            {
                "real": real_column.values,
                "fake": fake_column2.values,
                "feature": feature_column.values,
                "weight": weight_column.values,
            }
        )
        df2["bin"] = pd.cut(df2["feature"], bins=bins)
        group2 = df2.groupby("bin")


        # вычисляет эффективности и их отношения
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
        efficiencies2 = group2.apply(calculate_efficiencies_or_their_ratios, reverse=reverse)

        if raw_output_dict is not None:
            if 'GAN' not in raw_output_dict:
                raw_output_dict['GAN'] = {}
            if 'KDE' not in raw_output_dict:
                raw_output_dict['KDE'] = {}

            # assert name not in raw_output_dict['GAN'] or \
            #        name not in raw_output_dict['KDE']
            raw_output_dict['GAN'][name] = efficiencies.copy()
            raw_output_dict['KDE'][name] = efficiencies2.copy()
        figure = plt.figure(**self.figure_args)

        if self.make_ratio:
            plt.yscale("symlog", linthresh=1.0)
            # plt.grid(b=True, which="major", linewidth=1.25)
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

        # перебираем проценты
        for q in quantiles:
            if self.make_ratio:
                args = self.errorbar_common_args.copy()
                # надпись в легенду
                args["label"] = f'{q * 100}% ' + model_names[0]

                # Mokhnenko
                # цвет и маркер для данного процента
                # args['color'] = colors[q]
                # args['marker'] = markers[q]

                args['color'] = 'red'
                args['marker'] = 'o'

                # print('markeredgewidth:', args['markeredgewidth'])
                # берем эффективности нужного процента и вычитаем 1
                y_value = (efficiencies[f"eff_ratio_{q}"] - 1.0)
                # берем ошибки нужного процента
                yerr = efficiencies[
                    [f"eff_ratio_err_low_{q}", f"eff_ratio_err_high_{q}"]
                ].T.to_numpy()


                # рисует точки эффективности
                plt.errorbar(
                    x=efficiencies.index.categories.mid,
                    y=y_value,
                    xerr=(
                                 efficiencies.index.categories.right
                                 - efficiencies.index.categories.left
                         )
                         / 2,
                    yerr=yerr,
                    **args, # параметры
                    # {'fmt': 'o', 'marker': 'v', 'ms': 4, 'markeredgewidth': 2, 'label': '75.0% ', 'color': 'g'}
                )

                args["label"] = f'{q * 100}% ' + model_names[1]
                args['color'] = 'blue'
                args['marker'] = 's'
                y_value2 = (efficiencies2[f"eff_ratio_{q}"] - 1.0)
                # берем ошибки нужного процента
                yerr2 = efficiencies2[
                    [f"eff_ratio_err_low_{q}", f"eff_ratio_err_high_{q}"]
                ].T.to_numpy()

                plt.errorbar(
                    x=efficiencies2.index.categories.mid,
                    y=y_value2,
                    xerr=(
                                 efficiencies2.index.categories.right
                                 - efficiencies2.index.categories.left
                         )
                         / 2,
                    yerr=yerr2,
                    **args,  # параметры
                    # {'fmt': 'o', 'marker': 'v', 'ms': 4, 'markeredgewidth': 2, 'label': '75.0% ', 'color': 'g'}
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
        plt.text(0.05, 0.9, r'LHCb Simulation''\n'r'Preliminary', fontsize=30,
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
        ax.set_xlabel(x_label, fontweight='bold')
        ax.set_ylabel(y_label, fontweight='bold')
        plt.tight_layout(pad=3.0, w_pad=3.0, h_pad=0.2)
        plt.xticks()
        plt.yticks()

        return name, figure

    def make_figures(
            self, features, targets_real, targets_fake,
            targets_fake2, weights, raw_output_dict=None, model_names = []
    ):
        #
        if self.bins_2d is None:
            feature_columns_2d = [None]
        else:
            feature_columns_2d = features.columns

        # переюираем фичи
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

                # перебираем имена DLL-ок
                for target_column in targets_real.columns:
                    # перебираем имена фичей
                    for feature_column in features.columns:
                        if feature_column == feature_column_2d:
                            continue

                        # по умолчанию
                        # quantiles_list = [[0.95], [0.05], [0.95],
                        #                   [0.05]]
                        quantiles_list = self.thresholds


                        colors_list = [{0.75: 'g', 0.9: 'b', 0.95: 'r'}, {0.05: 'r', 0.1: 'b', 0.25: 'g'},
                                       {0.75: 'g', 0.9: 'b', 0.95: 'r'}, {0.05: 'r', 0.1: 'b', 0.25: 'g'}]
                        markers_list = [{0.75: 'v', 0.9: 's', 0.95: 'o'}, {0.05: 'o', 0.1: 's', 0.25: 'v'},
                                        {0.75: 'v', 0.9: 's', 0.95: 'o'}, {0.05: 'o', 0.1: 's', 0.25: 'v'}]
                        reverse_list = [False, False, True, True]

                        if self.make_ratio:
                            # цикл по 4 графикам где крайние проценты и реверс
                            for num in range(0, len(quantiles_list)):
                                yield self._make_efficiency_figure(
                                    real_column=targets_real[target_column].loc[sel],
                                    fake_column=targets_fake[target_column].loc[sel],
                                    fake_column2=targets_fake2[target_column].loc[sel],
                                    feature_column=features[feature_column].loc[sel],
                                    weight_column=weights.loc[sel],
                                    # quantiles=self.thresholds,
                                    quantiles=quantiles_list[num],
                                    name_suffix=name_suffix,
                                    title_suffix=title_suffix,
                                    raw_output_dict=raw_output_dict,
                                    colors=colors_list[num],
                                    markers=markers_list[num],
                                    reverse=reverse_list[num],
                                    model_names=model_names,
                                )


def plot_EfficiencyDouble(features, targets_real, targets_fake, targets_fake2,  weights, thresholds, model_names):
    # просто создает объект с параметрами по умолчанию
    plot_EfficiencyDouble = EfficiencyMakerDouble(
        period_in_epochs=None,
        bins=10,
        figure_args={'figsize': (8, 8)},
        errorbar_common_args={'fmt': 'o', 'marker': 'o', 'ms': 4, 'markeredgewidth': 1},
        errorbar_real_args={},
        errorbar_fake_args={},
        thresholds=thresholds,
        make_ratio=True,
        name_prefix='eff_ratio',
        bins_2d=None,
        per_bin_thresholds=True)

    figures_list  = []
    raw_output_dict = {'EfficiencyMaker:1': {} }
    figures_log_path = './compare_models'
    for name, figure in plot_EfficiencyDouble.make_figures(features, targets_real, targets_fake,
                                                           targets_fake2, weights, raw_output_dict['EfficiencyMaker:1'],
                                                           model_names=model_names):
        figures_list.append(figure)
       # figure.savefig(os.path.join(figures_log_path, f"{name}.pdf"))

    return  raw_output_dict