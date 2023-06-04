from pathlib import Path
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from .preprocessors import preprocessor_factory
from .factories import make_factory



class DataManager:
    def __init__(
        self,
        particle,
        data_path,
        data_shuffle_split_random_seed,
        test_size,
        target_columns,
        feature_columns,
        weight_column,
        preprocessor_config,
        preprocessor,
        extra_sample_config,
        csv_delimiter,
        preselection,
        **kwargs
    ):
        self.particle = particle
        self.data_path = data_path
        self.data_shuffle_split_random_seed = data_shuffle_split_random_seed
        self.test_size = test_size
        self.target_columns = target_columns
        self.feature_columns = feature_columns
        self.weight_column = weight_column
        self.preprocessor_config = preprocessor_config
        self.preprocessor = preprocessor
        self.csv_delimiter = csv_delimiter
        self.preselection = preselection

        if 'augmentations' in  kwargs:
            self.augmentations = kwargs['augmentations']
        else:
            self.augmentations = None



        self.columns = self.target_columns + self.feature_columns + [self.weight_column]
        self.filenames = list(Path(self.data_path).glob(f"{particle}_*.csv"))

        self.load_data()
        self.split_data()
        if extra_sample_config is not None:
            self.load_extra_sample(**extra_sample_config)

        if self.preprocessor is None:
            self.preprocessor = preprocessor_factory(**self.preprocessor_config)
            self.preprocessor.fit(self.data_train)
        elif self.preprocessor_config is not None:
            print(
                "Warning: ignoring preprocessor config, external preprocessor already provided"
            )

        ### replace `self.data_extra` with `self.data_extra`
        ### replace `self.data_train` with `self.data_train`
        ### replace `self.data_train` with `self.data_train`

        #if regularization is not None:
            #regularization_function(self)
            #print(self.data)
            #print("Nothing is Broken")
            #print("Smth")

    def load_data(
        self,
        extra=False,
        particle=None,
        path=None,
        csv_delimiter=None,
        target_columns=None,
        feature_columns=None,
        weight_column=None,
        aux_features=None,
    ):
        assert (
            (target_columns is None)
            == (feature_columns is None)
            == (weight_column is None)
        )
        if aux_features is None:
            aux_features = []
        elif not extra:
            raise NotImplementedError(
                "Aux features functionality only works for extra sample so far."
            )
            # TODO: implement aux features for the main data as well (if ever needed (?)).

        if csv_delimiter is None:
            csv_delimiter = self.csv_delimiter
        if extra:
            assert particle is not None
            if path is None:
                path = self.data_path
            filenames = list(Path(path).glob(f"{particle}_*.csv"))
        else:
            filenames = self.filenames

            if self.augmentations != None:
                # loading augmentations from list
                aug_list = self.augmentations.split(',')
                for augmentation in aug_list:
                    aug_path = os.path.join(self.data_path, augmentation + '.csv')
                    if os.path.exists(aug_path):
                        filenames.append(Path(aug_path))
                    else:
                        print('Augmentation not found:', augmentation)


        assert len(filenames) > 0
        print(f"Loading {'extra' if extra else 'main'} dataset", flush=True)
        if target_columns is None:
            columns = self.columns
        else:
            columns = target_columns + feature_columns + [weight_column]

        aux_features = [f for f in aux_features if f not in columns]

        def transverse_momentum(p, eta):
            p_t = abs(p) / np.cosh(eta)
            return p_t

        def load_and_concat():
            print(columns)
            print(filenames)
            if 'P_T' in columns:
                columns.remove('P_T')
            loaded_data = pd.concat(
                [
                    pd.read_csv(
                        fname, delimiter=csv_delimiter, usecols=columns + aux_features
                    )
                    for fname in tqdm(filenames)
                ],
                axis=0,
                ignore_index=True,
            )
            if 'P_T' in self.feature_columns:
                # Add feature
                columns.append('P_T')
                if 'Brunel_P' in list(loaded_data.columns.values):
                    P_T = transverse_momentum(loaded_data['Brunel_P'], loaded_data['Brunel_ETA'])
                else:
                    P_T = transverse_momentum(loaded_data['P'], loaded_data['ETA'])
                loaded_data['P_T'] = P_T

            if self.preselection is not None:
                loaded_data = loaded_data.loc[loaded_data.eval(self.preselection)]
            return loaded_data[columns], loaded_data[aux_features]

        try:
            loaded_data, aux_data = load_and_concat()
        except ValueError:
            # re-trying without the weight column:
            # weight_column = columns[-1]
            # columns = columns[:-1]

            if 'probe_sWeight' in columns:
                columns.remove('probe_sWeight')
            loaded_data, aux_data = load_and_concat()
            loaded_data['probe_sWeight'] = 1.0

            if 'P_T' in self.feature_columns:
                # crutch way to set the correct order of the columns
                loaded_data = loaded_data.loc[:, ['RichDLLe', 'RichDLLk', 'RichDLLmu',
                                                  'RichDLLp', 'RichDLLbt', 'P', 'ETA',
                                                  'NumSPDHits',  'probe_sWeight', 'P_T']]
            else:
                loaded_data = loaded_data.loc[:, ['RichDLLe', 'RichDLLk', 'RichDLLmu',
                                                  'RichDLLp', 'RichDLLbt', 'P', 'ETA',
                                                  'NumSPDHits', 'probe_sWeight']]
            columns = loaded_data.columns.values.tolist()



            print(
                f"Warning: coulnd't find the weight column ({weight_column})."
                " Setting all weights to 1."
            )
        if columns != self.columns:
            assert len(columns) == len(self.columns)
            # here we do feature renaming
            loaded_data.columns = self.columns
        if extra:
            self.data_extra = loaded_data
            if not aux_data.empty:
                self.data_extra_aux = aux_data
        else:
            self.data = loaded_data



    def load_extra_sample(self, **kwargs):
        self.load_data(extra=True, **kwargs)

    def split_data(self):
        self.data_train, self.data_val = train_test_split(
            self.data,
            test_size=self.test_size,
            random_state=self.data_shuffle_split_random_seed,
        )

        self.data_val, self.data_test = train_test_split(
            self.data_val,
            test_size=self.test_size,
            random_state=self.data_shuffle_split_random_seed,
        )


    def get_preprocessed_data(self, split, with_aux=False):
        if not hasattr(self, "_preprocessing_cache"):
            self._preprocessing_cache = {}

        if split in self._preprocessing_cache:
            return self._preprocessing_cache[split]

        if split == "train":
            data = self.data_train
        elif split == "val":
            data = self.data_val
        elif split == "test":
            data = self.data_test
        elif split == "extra":
            data = self.data_extra
        else:
            raise NotImplementedError(f"split={split}")

        data = self.preprocessor.transform(data)

        # далаем все веса одинаковыми
        data = data.assign(probe_sWeight=1.0)
        self._preprocessing_cache[split] = data

        if with_aux:
            assert split == "extra", "Aux features only implemented for the extra split"
            return data, self.data_extra_aux
        return data

    def get_batch_generator(self, batch_size, split="train", shuffle=True, drop_P_T=False):
        data = self.get_preprocessed_data(split)
        if drop_P_T:
            if 'P_T' in data.columns.values.tolist():
                data = data.drop(columns=['P_T'])
            if 'P_T' in self.feature_columns:
                self.feature_columns.remove('P_T')
        if shuffle:
            ids = np.random.choice(len(data), size=len(data), replace=False)
            data = data.iloc[ids]

        for ii in range(0, len(data), batch_size):
            yield data.iloc[ii : ii + batch_size]



dm_factory = make_factory([DataManager])

def create_data_manager(**kwargs):
    return dm_factory(classname="DataManager", **kwargs)

