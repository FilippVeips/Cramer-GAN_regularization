class_constructor_defaults = {}


####################################################################################
####################################################################################
############## GAN

class_constructor_defaults["GANUpdaterBase"] = dict(
    gen_lr=0.001,
    disc_lr=0.001,
    gen_optimizer="RMSprop",
    disc_optimizer="RMSprop",
    gen_lr_scheduler={"classname": "ExponentialDecayScheduler"},
    disc_lr_scheduler={"classname": "ExponentialDecayScheduler"},
)

class_constructor_defaults["CramerUpdater"] = dict(
    gp_lambda=0.0,
    gp_lambda_scheduler={"classname": "ArctanSaturatingScheduler"},
    **class_constructor_defaults["GANUpdaterBase"]
)

class_constructor_defaults["WGANUpdater"] = dict(
    gp_lambda=0.0,
    gp_lambda_scheduler={"classname": "ArctanSaturatingScheduler"},
    **class_constructor_defaults["GANUpdaterBase"]
)

class_constructor_defaults["SimpleGenerator"] = dict(
    n_latent_dims=64,
    distribution="normal",
    depth=10,
    width=128,
    activation="relu",
    input_size=3,
    output_size=5,
    arch_name="SimpleGenerator",
)

class_constructor_defaults["SimpleDiscriminator"] = dict(
    depth=10,
    width=128,
    activation="relu",
    input_size_main=5,
    input_size_cond=3,
    output_size=256,
    arch_name="SimpleDiscriminator",
)

class_constructor_defaults["GANModel"] = dict(
    generator_config={"classname": "SimpleGenerator"},
    discriminator_config={"classname": "SimpleDiscriminator"},
    updater_config={"classname": "CramerUpdater"},
    step_scheduler_config={"classname": "KStepScheduler"},
)

class_constructor_defaults["KStepScheduler"] = dict(k=5)


####################################################################################
####################################################################################
############## Variable schedulers

class_constructor_defaults["ExponentialDecayScheduler"] = dict(
    decay_steps=10, decay_rate=0.98
)

class_constructor_defaults["ArctanSaturatingScheduler"] = dict(
    magnitude=20.0, halfrise_steps=500, force_value_at_start=True, minimal_value=0.02
)


####################################################################################
####################################################################################
############## Training

class_constructor_defaults["TrainingManager"] = dict(
    batch_size=50000,
    epochs=5000,
    log_path="logs",
    save_base_path="saved_models",
    save_interval_in_epochs=100,
)


####################################################################################
####################################################################################
############## Data

class_constructor_defaults["DataManager"] = dict(
    data_path="/home/amaevskiy/temporary/data_calibsample/",
    data_shuffle_split_random_seed=123,
    test_size=0.5,
    target_columns=["RichDLLe", "RichDLLk", "RichDLLmu", "RichDLLp", "RichDLLbt"],
    feature_columns=["Brunel_P", "Brunel_ETA", "nTracks_Brunel"],
    weight_column="probe_sWeight",
    preprocessor_config=dict(
        classname="WeightBypassPreprocessor",
        weight_col_name="probe_sWeight",
        preprocessor_config=dict(
            classname="QuantileTransformer",
            output_distribution="normal",
            n_quantiles=100000,
            subsample=10000000000,
        ),
    ),
    preprocessor=None,
    extra_sample_config=None,
    csv_delimiter="\t",
    preselection=None,
)

class_constructor_defaults["WeightBypassPreprocessor"] = dict(
    preprocessor=None, preprocessor_config=None
)


####################################################################################
####################################################################################
############## Metrics

class_constructor_defaults["SummaryMetricsMaker"] = dict(
    split="val",
    period_in_epochs=100,
    postprocess=True,
    plot_maker_configs=[
        dict(classname="Hist1DMaker"),
        dict(classname="EfficiencyMaker"),
    ],
    scalar_maker_configs=[dict(classname="WeightedKSMaker")],
    selection=None,
    aux_features_in_selection=False,
    selection_augmentation=[],
    figures_log_path=None,
    accept_reject_gen_config=None,
)

class_constructor_defaults["WeightedKSMaker"] = dict(bins=20, period_in_epochs=None)

class_constructor_defaults["Hist1DMaker"] = dict(
    period_in_epochs=None,
    bins=100,
    figure_args=dict(figsize=(8, 8)),
    hist_common_args=dict(density=True),
    hist_real_args=dict(label=r"detailed" "\n" r"simulation"),
    hist_fake_args=dict(label="GAN", alpha=0.7),
    name_prefix="hist1d",
    logy=False,
)

class_constructor_defaults["EfficiencyMaker"] = dict(
    period_in_epochs=None,
    bins=10,
    figure_args=dict(figsize=(8, 8)),
    errorbar_common_args=dict(fmt="o", marker="o", ms=4, markeredgewidth=2),
    errorbar_real_args=dict(),
    errorbar_fake_args=dict(),
    thresholds=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
    make_ratio=True,
    name_prefix="eff_ratio",
    bins_2d=None,
    per_bin_thresholds=False,
)

class_constructor_defaults["ProbNNAugmentation"] = dict(feature_mapping=None)
