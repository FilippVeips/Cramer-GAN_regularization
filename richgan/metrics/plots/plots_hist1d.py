import numpy as np
from .mpl_setup import plt
from .plots_base import PlotMakerBase


class Hist1DMaker(PlotMakerBase):
    def __init__(
        self,
        period_in_epochs,
        bins,
        figure_args,
        hist_common_args,
        hist_real_args,
        hist_fake_args,
        name_prefix,
        logy,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.period_in_epochs = period_in_epochs
        self.bins = bins
        self.figure_args = figure_args
        self.hist_common_args = hist_common_args
        self.hist_real_args = hist_real_args
        self.hist_fake_args = hist_fake_args
        self.name_prefix = name_prefix
        self.logy = logy

    def make_hist_figure(self, real_column, fake_column, weight_column, title):
        bins = np.linspace(
            min(real_column.min(), fake_column.min()),
            max(real_column.max(), fake_column.max()),
            self.bins + 1,
        )

        # Mokhnenko
        # bins = 50
        binwidth = 10


        self.hist_fake_args['label'] = 'KDE'

        self.hist_fake_args['histtype'] = 'step'
        self.hist_real_args['histtype'] = 'stepfilled'
        self.hist_fake_args['color'] = '#FF0000'
        self.hist_real_args['color'] = '#0000FF'
        self.hist_fake_args['linewidth'] = 3
        self.hist_fake_args['alpha'] = 1
        self.hist_real_args['alpha'] = 0.6
        bins = np.arange(min(real_column.min(), fake_column.min()), max(real_column.max(), fake_column.max()) + binwidth, binwidth)
        figure = plt.figure(**self.figure_args)
        plt.hist(
            real_column,
            bins=bins,
            weights=weight_column,
            **self.hist_common_args,
            **self.hist_real_args,
        )


        plt.hist(
            fake_column,
            bins=bins,
            weights=weight_column,
            **self.hist_common_args,
            **self.hist_fake_args,
        )
        if self.logy:
            plt.yscale("log")

        if title =='RichDLLe':
            plt.xlim([-150, 50])


        ax = plt.gca()
        plt.text(0.05, 0.90, r'LHCb Simulation''\n'r'Preliminary', fontsize=30,
                 verticalalignment='center', transform=ax.transAxes)
        ax.set_ylabel('a.u.', fontweight='bold', fontsize=32)
        ax.set_xlabel(title, fontweight='bold', fontsize=32)
        plt.legend(loc=(0.1, 0.56))
        # plt.title(title, fontsize=32, fontweight='bold', y=-0.05)
        # plt.title(title)

        return figure

    def make_figures(
        self, features, targets_real, targets_fake, weights, raw_output_dict=None
    ):
        for column in targets_real.columns:
            figure = self.make_hist_figure(
                real_column=targets_real[column],
                fake_column=targets_fake[column],
                weight_column=weights,
                title=column,
            )
            yield f"{self.name_prefix}_{column}", figure
