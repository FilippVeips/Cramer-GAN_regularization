from ...utils.factories import make_factory
from .plots_hist1d import Hist1DMaker
from .plots_efficiency import EfficiencyMaker


plot_maker_factory = make_factory([Hist1DMaker, EfficiencyMaker])
