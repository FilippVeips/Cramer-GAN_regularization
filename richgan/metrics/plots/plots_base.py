class PlotMakerBase:
    def __init__(self):
        self.period_in_epochs = None

    def make_figures(
        self, features, targets_real, targets_fake, weights, raw_output_dict=None
    ):
        raise NotImplementedError("Re-implement this method in a sub-class.")
