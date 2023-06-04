from .factories import make_factory
from ..ext.probnn.probnn import MC15TuneV1


class AugmentationBase:
    def augment(self, data):
        raise NotImplementedError("Implement this method in a sub-class")


class ProbNNAugmentation(AugmentationBase):
    def __init__(
        self, config_path, particle_name, target_feature_name, feature_mapping
    ):
        self.probnn = MC15TuneV1(config_path, particle_name)
        self.target_feature_name = target_feature_name
        self.feature_mapping = feature_mapping

    def augment(self, data):
        original_data = data
        if self.feature_mapping is not None:
            data = data.copy()
            for src_feature, target_feature in self.feature_mapping.items():
                data[target_feature] = data[src_feature]
        assert self.target_feature_name not in original_data.columns
        original_data[self.target_feature_name] = self.probnn.predict(data)


aug_factory = make_factory([ProbNNAugmentation])
