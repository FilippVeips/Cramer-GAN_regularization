import io
import numpy as np
import pandas as pd
import tensorflow as tf
import PIL
from pathlib import Path
import pickle
from .plots.mpl_setup import plt
from .plots import plot_maker_factory
from .scalars import scalar_maker_factory
from ..utils.lazy import LazyCall
from ..utils.factories import make_factory
from ..utils.event_schedulers import global_check_step
from ..utils.feature_augmentation import aug_factory


def figure_to_img_data(figure, close_figure):
    buf = io.BytesIO()
    figure.savefig(buf, format="png")
    if close_figure:
        plt.close(figure)
    buf.seek(0)
    img = PIL.Image.open(buf)
    return np.array(img.getdata(), dtype=np.uint8).reshape(
        1, img.size[1], img.size[0], -1
    )


class SummaryMetricsMaker:
    """
    A class for handling the summary metrics collection and recording.

    This class is an intermediate layer between the training manager and individual plot- and
    scalar-summary makers. It collects the summary makers together and runs them when its
    summary_callback method is being called by the training manager. It then handles dumping the
    summary makers' outputs to tensorboard, pdf and raw pickled data.

    List of public methods:
      - summary_callback
    """

    def __init__(
        self,
        summary_writer,
        model,
        data_manager,
        split,
        period_in_epochs,
        postprocess,
        plot_maker_configs,
        scalar_maker_configs,
        selection,
        aux_features_in_selection,
        selection_augmentation,
        figures_log_path,
        accept_reject_gen_config,
        estimator,

    ):
        """
        Constructor.

        Arguments:
        summary_writer -- TensorFlow SummaryWriter object
        model -- a GANModel instance
        data_manager -- a DataManager instance
        split -- a string specifying which data_manager's split to use (train/test/val/extra)
        period_in_epochs -- how often to produce the summaries (in epochs)
        postprocess -- boolean flag, whether the generated and real data should be
            postprocessed by the data_manager's inverse_transform method
        plot_maker_configs -- list of dictionary configs for creating plot makers
            (see PlotMakerBase)
        scalar_maker_configs -- list of dictionary configs for creating scalar makers
            (see ScalarMakerBase)
        selection -- a string to be evaluated with pandas.DataFrame.eval that should return a
            boolean selection mask. The generated targets are prefixed with 'gen_'.
        aux_features_in_selection -- a boolean flag denoting whether the selection string depends
            on the data_manager's aux features
        selection_augmentation -- list of dictionary configs to create augmentations (see
            AugmentationBase). These may only be used in the selection string.
        figures_log_path -- None or a string specifying where to export the figures to
        accept_reject_gen_config -- None or a dictionary config to run the accept-reject
            selection procedure. This is an alternative to the selection string parameter above.
            In the accept-reject mode the GAN is run multiple times until all the events it has
            generated satisfy the selection specified in this config.
            See richgan/configs/simple.mc/eval.on.Lb-accrej.yaml for an example configuration.
        """
        self.summary_writer = summary_writer
        self.model = model
        self.data_manager = data_manager
        self.split = split
        self.period_in_epochs = period_in_epochs
        self.postprocess = postprocess
        self.estimator = estimator

        self.plot_makers = [
            plot_maker_factory(**kwargs) for kwargs in plot_maker_configs
        ]
        self.scalar_makers = [
            scalar_maker_factory(**kwargs) for kwargs in scalar_maker_configs
        ]

        self.selection = selection
        self.aux_features_in_selection = aux_features_in_selection
        self.selection_augmentation = [
            aug_factory(**kwargs) for kwargs in selection_augmentation
        ]
        self.figures_log_path = figures_log_path
        self.accept_reject_gen_config = accept_reject_gen_config

    def summary_callback(self, global_step):
        if not global_check_step(global_step, self.period_in_epochs):
            return

        data_dict = LazyCall(self._get_data)

        if self.figures_log_path is not None:
            raw_output_dict = {}
            figures_log_path = Path(self.figures_log_path) / f"{global_step:06d}"

        for i_pm, plot_maker in enumerate(self.plot_makers):
            make_figures_additional_args = {}
            if self.figures_log_path is not None:
                key = f"{plot_maker.__class__.__name__}:{i_pm}"
                assert key not in raw_output_dict
                raw_output_dict[key] = {}
                make_figures_additional_args["raw_output_dict"] = raw_output_dict[key]
            if plot_maker.period_in_epochs is not None:
                if not global_check_step(global_step, plot_maker.period_in_epochs):
                    continue
            for name, figure in plot_maker.make_figures(
                **data_dict(),
                **make_figures_additional_args,
            ):
                if self.figures_log_path is not None:
                    figures_log_path.mkdir(parents=True, exist_ok=True)
                    figure.savefig(figures_log_path / f"{name}.pdf")
                self._write_figure(name, figure, global_step)


        if self.figures_log_path is not None:
            with open(figures_log_path / "raw_output_dict_KDE.pkl", "wb") as f:
                pickle.dump(raw_output_dict, f)
        for scalar_maker in self.scalar_makers:
            if scalar_maker.period_in_epochs is not None:
                if not global_check_step(global_step, scalar_maker.period_in_epochs):
                    continue
            for name, value in scalar_maker.make_scalars(**data_dict()):
                self._write_scalar(name, value, global_step)

    def _generate_targets(self, features, targets_real):
        if 'P_T' in features.columns.values.tolist():
            features2 = features.drop(columns=['P_T'])
        else:
            features2 = features
        if self.accept_reject_gen_config is None:
            return pd.DataFrame(
                self.model.generator(
                    tf.convert_to_tensor(features2, dtype="float32"), training=False
                ).numpy(),
                columns=targets_real.columns,
                index=targets_real.index,
            )

        max_iterations = self.accept_reject_gen_config.get("max_iterations", 10)
        print(
            f"SummaryMetricsMaker: running in accept-reject mode (max_iterations={max_iterations})."
        )
        cuts = pd.DataFrame()
        for column, cut in self.accept_reject_gen_config["cuts"].items():
            for cut_type, cut_value in cut.items():
                cuts.loc[cut_type, column] = cut_value
        if "lower" not in cuts.index:
            cuts.loc["lower"] = np.nan
        if "upper" not in cuts.index:
            cuts.loc["upper"] = np.nan
        for column in self.data_manager.columns:
            if column not in cuts.columns:
                cuts[column] = np.nan
        assert cuts.shape == (2, len(self.data_manager.columns))
        print("SummaryMetricsMaker: the accept-reject cuts are:")
        print(cuts)

        cuts_processed = self.data_manager.preprocessor.transform(
            cuts[self.data_manager.columns]
        )
        print(cuts_processed)
        lower = cuts_processed.loc["lower"]
        upper = cuts_processed.loc["upper"]

        ids_to_generate_for = features.index.copy()
        generated_targets = pd.DataFrame()
        non_target_columns = [
            col for col in self.data_manager.columns if col not in targets_real.columns
        ]
        while len(ids_to_generate_for) > 0:
            max_iterations -= 1
            assert max_iterations >= 0
            print(
                f"SummaryMetricsMaker: events to generate left: {len(ids_to_generate_for)}"
            )
            gen_i = pd.DataFrame(
                self.model.generator(
                    tf.convert_to_tensor(
                        features.loc[ids_to_generate_for], dtype="float32"
                    ),
                    training=False,
                ).numpy(),
                columns=targets_real.columns,
                index=ids_to_generate_for,
            )
            gen_i[non_target_columns] = np.nan

            selection = (lower.isna() | (gen_i >= lower)).all(axis=1) & (
                upper.isna() | (gen_i <= upper)
            ).all(axis=1)
            gen_i = gen_i.loc[selection]
            generated_targets = pd.concat(
                [generated_targets, gen_i[targets_real.columns]], axis=0
            )

            ids_to_generate_for = ids_to_generate_for.difference(gen_i.index)

        print("SummaryMetricsMaker: Done accept-reject generation")

        return generated_targets.loc[features.index]

    def _generate_targets_kde(self, train_features, train_weights, train_targets, features, targets_real):
        from kde_baseline.conditional_kde import ConditionalKDE
        kde = ConditionalKDE(bandwidth=float(self.estimator['bandwidth'])).fit(
            train_targets.to_numpy(), train_weights.to_numpy(), train_features.to_numpy())
        # gen_test_target, weights_fake = kde.sample(features.to_numpy(),
        gen_test_target = kde.sample(features.to_numpy(),
                                     int(self.estimator['sample_bw_factor']),
                                     int(self.estimator['neighbors']),
                                     progress=True)
        # to pandas
        gen_test_target = pd.DataFrame(gen_test_target, columns=self.data_manager.target_columns)

        # weights_fake = pd.DataFrame(weights_fake, columns=[self.data_manager.weight_column])
        # return gen_test_target, weights_fake
        return gen_test_target

    def _get_data(self):
        data_real = self.data_manager.get_preprocessed_data(
            split=self.split, with_aux=self.aux_features_in_selection
        )
        # FIXME: Undoing the separation
        # test_data = self.data_manager.get_preprocessed_data(split="test")
        # data_real = pd.concat([data_real, test_data], ignore_index=True)

        data_real = data_real.reset_index(drop=True)

        if str(self.estimator['sample_count']) == 'all':
            pass
        else:
            data_real = data_real[: int(self.estimator['sample_count'])]
        print('Test dataset size:', len(data_real))
        if self.aux_features_in_selection:
            data_real, data_real_aux = data_real
        features = data_real[self.data_manager.feature_columns]
        targets_real = data_real[self.data_manager.target_columns]

        if self.estimator['model'] == 'KDE':
            # For KDE, we also need a training dataset.
            train_data = self.data_manager.get_preprocessed_data(split="train").reset_index(drop=True)
            train_features = train_data[self.data_manager.feature_columns]
            train_targets = train_data[self.data_manager.target_columns]
            train_weights = train_data[self.data_manager.weight_column]

            # targets_fake, weights_fake = self._generate_targets_kde(train_features, train_weights, train_targets,
            #                                             features, targets_real)

            targets_fake = self._generate_targets_kde(train_features, train_weights, train_targets,
                                                                    features, targets_real)


            # Only debug!
            # targets_fake = self._generate_targets_kde(train_features, train_targets,
            #                                           train_features, targets_real)
            # check with the same piece!
            #  data_real = train_data
            targets_fake = targets_fake.reset_index(drop=True)
            # weights_fake = weights_fake.reset_index(drop=True)
            """
            with open('dataset.pickle', 'wb') as f:
                data = {
                    'val': data_real,
                    'train': train_data,
                    'fake': targets_fake,
                }
                pickle.dump(data, f)
            """
        elif self.estimator['model'] == 'GANvsKDE':

            #KDE
            train_data = self.data_manager.get_preprocessed_data(split="train").reset_index(drop=True)
            train_features = train_data[self.data_manager.feature_columns]
            train_targets = train_data[self.data_manager.target_columns]
            train_weights = train_data[self.data_manager.weight_column]
            targets_fake_kde = self._generate_targets_kde(train_features, train_weights, train_targets,
                                                      features, targets_real)
            targets_fake_kde = targets_fake_kde.reset_index(drop=True)

            #GAN
            targets_fake_gan = self._generate_targets(features, targets_real)
            targets_fake_gan = targets_fake_gan.reset_index(drop=True)


            weights = data_real[self.data_manager.weight_column]
            if self.postprocess:
                data_fake_kde  = pd.concat([targets_fake_kde, features, weights], axis=1)[data_real.columns]
                data_fake_gan = pd.concat([targets_fake_gan, features, weights], axis=1)[data_real.columns]

                data_real_postprocessed = self.data_manager.preprocessor.inverse_transform(
                    data_real
                )
                data_fake_kde_postprocessed = self.data_manager.preprocessor.inverse_transform(
                    data_fake_kde
                )
                data_fake_gan_postprocessed = self.data_manager.preprocessor.inverse_transform(
                    data_fake_gan
                )
                features = data_real_postprocessed[self.data_manager.feature_columns]
                targets_real = data_real_postprocessed[self.data_manager.target_columns]

                targets_fake_kde = data_fake_kde_postprocessed[self.data_manager.target_columns]
                targets_fake_gan = data_fake_gan_postprocessed[self.data_manager.target_columns]

            if self.selection is not None:
                print("Applying selection:", self.selection)
                targets_fake_copy_kde = targets_fake_kde.copy()
                targets_fake_copy_kde.columns = "gen_" + targets_fake_copy_kde.columns

                targets_fake_copy_gan = targets_fake_gan.copy()
                targets_fake_copy_gan.columns = "gen2_" + targets_fake_copy_gan.columns

                selection_ds = pd.concat(
                    [features, targets_real, targets_fake_copy_kde, targets_fake_copy_gan], axis=1
                )
                if self.aux_features_in_selection:
                    selection_ds = pd.concat([selection_ds, data_real_aux], axis=1)
                for augmentation in self.selection_augmentation:
                    augmentation.augment(selection_ds)

                selection_mask = selection_ds.eval(self.selection)
                print(f"Selection keeps {selection_mask.mean() * 100}% of events")

                features = features.loc[selection_mask]
                targets_real = targets_real.loc[selection_mask]

                targets_fake_kde = targets_fake_kde.loc[selection_mask]
                targets_fake_gan = targets_fake_gan.loc[selection_mask]
                weights = weights.loc[selection_mask]

            targets_data_dict = {
                'features': features,
                'targets_real': targets_real,
                'weights': weights,
                'targets_fake_kde': targets_fake_kde,
                'targets_fake_gan': targets_fake_gan
            }
            with open("targets_data_dict.pkl", "wb") as f:
                pickle.dump(targets_data_dict, f)
            exit()

        else:
            targets_fake = self._generate_targets(features, targets_real)
            targets_fake = targets_fake.reset_index(drop=True)


        # FIXME:
        # data_real = data_real.assign(probe_sWeight=1.0)
        weights = data_real[self.data_manager.weight_column]

        if self.postprocess:
            if self.estimator['model'] == 'KDE':
                # data_fake = pd.concat([targets_fake, features, weights_fake], axis=1)[data_real.columns]
                data_fake = pd.concat([targets_fake, features, weights], axis=1)[data_real.columns]
            else:

                print('features:', features)
                print('data_real.columns:', data_real.columns)
                if 'P_T' in features.columns.values.tolist():
                    features2 = features.drop(columns=['P_T'])
                else:
                    features2 = features

                if 'P_T' in data_real.columns.values.tolist():
                    data_real2 = data_real.drop(columns=['P_T'])
                else:
                    data_real2 = data_real

                data_fake = pd.concat([targets_fake, features2, weights], axis=1)[data_real2.columns]

            data_real_postprocessed = self.data_manager.preprocessor.inverse_transform(
                data_real
            )
            data_fake_postprocessed = self.data_manager.preprocessor.inverse_transform(
                data_fake
            )

            features = data_real_postprocessed[self.data_manager.feature_columns]
            targets_real = data_real_postprocessed[self.data_manager.target_columns]
            targets_fake = data_fake_postprocessed[self.data_manager.target_columns]

        if self.selection is not None:
            print("Applying selection:", self.selection)
            targets_fake_copy = targets_fake.copy()
            targets_fake_copy.columns = "gen_" + targets_fake_copy.columns
            selection_ds = pd.concat(
                [features, targets_real, targets_fake_copy], axis=1
            )
            if self.aux_features_in_selection:
                selection_ds = pd.concat([selection_ds, data_real_aux], axis=1)
            for augmentation in self.selection_augmentation:
                augmentation.augment(selection_ds)

            selection_mask = selection_ds.eval(self.selection)
            print(f"Selection keeps {selection_mask.mean() * 100}% of events")

            features = features.loc[selection_mask]
            targets_real = targets_real.loc[selection_mask]
            targets_fake = targets_fake.loc[selection_mask]
            weights = weights.loc[selection_mask]

        return dict(
            features=features,
            targets_real=targets_real,
            targets_fake=targets_fake,
            weights=weights,
        )

    def _write_figure(self, name, figure, global_step, close_figure=True):
        with self.summary_writer.as_default():
            tf.summary.image(
                f"{name}/{self.split}",
                figure_to_img_data(figure, close_figure=close_figure),
                global_step,
            )

    def _write_scalar(self, name, value, global_step):
        with self.summary_writer.as_default():
            tf.summary.scalar(f"{name}/{self.split}", value, global_step)


smm_factory = make_factory([SummaryMetricsMaker])


def create_summary_maker(**kwargs):
    return smm_factory(classname="SummaryMetricsMaker", **kwargs)
