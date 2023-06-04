from sklearn import preprocessing as sk_preps
import pandas as pd
from .factories import make_factory


class WeightBypassPreprocessor:
    def __init__(self, weight_col_name, preprocessor, preprocessor_config):
        assert (preprocessor is None) != (preprocessor_config is None)
        if preprocessor_config is not None:
            self.preprocessor = preprocessor_factory(**preprocessor_config)
        else:
            self.preprocessor = preprocessor
        self.weight_col_name = weight_col_name

    def strip_weights(self, data):
        return data.drop(self.weight_col_name, axis=1)

    def fit(self, data):
        self.preprocessor.fit(self.strip_weights(data))
        return self

    def transform(self, data, inverse=False):
        transform_func = (
            self.preprocessor.inverse_transform
            if inverse
            else self.preprocessor.transform
        )

        data_without_weights = self.strip_weights(data)
        transformed = transform_func(data_without_weights)
        return pd.concat(
            [
                pd.DataFrame(
                    transformed,
                    columns=data_without_weights.columns,
                    index=data_without_weights.index,
                ),
                data[self.weight_col_name],
            ],
            axis=1,
        )[data.columns]

    def inverse_transform(self, data):
        return self.transform(data, inverse=True)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


preprocessor_factory = make_factory(
    [getattr(sk_preps, classname) for classname in sk_preps.__all__]
    + [WeightBypassPreprocessor]
)
