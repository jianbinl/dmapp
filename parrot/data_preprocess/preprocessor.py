# coding = utf-8

import pandas as pd


def check_data(data):
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    cols = data.columns
    for col in cols:
        data[col] = pd.to_numeric(data[col], errors='ignore')
    return data


def get_object_features(data):
    cols = data.columns
    fea = [c for c in cols if data[c].dtype == 'object']
    return data[fea]


def get_numeric_features(data, errors='ignore'):
    assert errors in ['ignore', 'coerce']
    cols = data.columns
    new_data = data.copy()
    for col in cols:
        try:
            pd.to_numeric(data[col], errors='raise')
            new_data[col] = pd.to_numeric(new_data[col], errors=errors)
        except ValueError:
            new_data.drop(col, axis=1, inplace=True)
    return new_data


class DataPreprocessor:
    """
    Data preprocess dataset
    """

    def __init__(self, data):
        self.data = check_data(data)

    def quick_preprocess(self, drop_na_features=False, feature_na_ratio=None,
                         drop_na_samples=False, sample_na_ratio=None) -> pd.DataFrame:
        """
        """
        print(f"{'*' * 50} Start Preprocessing {'*' * 45}")
        self.overview()
        x = self.drop_features_unchanged(self.data)
        if drop_na_features:
            x = self.drop_features_na(self.data, ratio=feature_na_ratio)
        if drop_na_samples:
            x = self.drop_samples_na(self.data, ratio=sample_na_ratio)
        print(f"{'*' * 50} End Preprocess {'*' * 50}")
        return x

    def overview(self) -> None:
        self.data_shape = self.data.shape
        self.object_features = [c for c in self.data.columns if self.data[c].dtype == 'object']
        self.numeric_features = [c for c in self.data.columns if self.data[c].dtype
                                 in ['float64', 'int64', 'float32', 'int32']]
        print(f'===== Data overview: \n'
              f'data shape: {self.data_shape}\n'
              f'object features: {self.object_features}\n'
              f'numeric features: {self.numeric_features}')

    @staticmethod
    def drop_features_unchanged(x: pd.DataFrame) -> pd.DataFrame:
        """drop unchanged columns"""
        cols = x.columns
        dlt = []
        for col in cols:
            cnt = x[col].value_counts(dropna=False).shape[0]
            if cnt == 1:
                dlt.append(col)
        xp = x.drop(dlt, axis=1)
        print(f"===== Dropping features unchanged : \n{dlt}")
        return xp

    @staticmethod
    def drop_features_na(x: pd.DataFrame, ratio=None) -> pd.DataFrame:
        """drop na ratio > ratio columns"""
        if ratio is None:
            ratio = 0.
        thresh = round(x.shape[0] * (1 - ratio), 0)
        xp = x.dropna(axis=1, thresh=thresh)
        dlt = set(x.columns) - set(xp.columns)
        print(f"===== Dropping features the percentage of null values is > {ratio}: \n{dlt}")
        return xp

    @staticmethod
    def drop_samples_na(x: pd.DataFrame, ratio=None) -> pd.DataFrame:
        if ratio is None:
            ratio = 0.
        thresh = round(x.shape[1] * (1 - ratio), 0)
        xp = x.dropna(axis=0, thresh=thresh)
        dlt = set(x.index) - set(xp.index)
        print(f"===== Dropping samples the percentage of null values is > {ratio}: \n{dlt}")
        return xp
