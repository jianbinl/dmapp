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
    """
    Return a new dataframe just only contain categories features

    Parameters
    ----------
    data: pd.DataFrame

    Returns
    -------
    new_data: pd.DataFrame
        A new dataframe just only contain columns that dtype is `object`
    """
    cols = data.columns
    fea = [c for c in cols if data[c].dtype == 'object']
    return data[fea]


def get_numeric_features(data, errors='ignore'):
    """
    Return a new dataframe just only contain numeric features

    Parameters
    ----------
    data: pd.DataFrame
    errors: str
        ['ignore', 'coerce']

    Returns
    -------
    new_data: pd.DataFrame
    """
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

    def __init__(self):
        ...

    @classmethod
    def overview(cls, x: pd.DataFrame) -> None:
        data_shape = x.shape
        object_features = [c for c in x.columns if x[c].dtype == 'object']
        numeric_features = [c for c in x.columns if x[c].dtype
                                 in ['float64', 'int64', 'float32', 'int32']]
        print(f'===== Data overview: \n'
              f'data shape: {data_shape}\n'
              f'object features: {object_features}\n'
              f'numeric features: {numeric_features}')

    @staticmethod
    def drop_features_unchanged(x: pd.DataFrame) -> pd.DataFrame:
        """
        Drop columns that value unchanged

        Parameters
        ----------
        x: pd.DataFrame
            A dataframe

        Returns
        -------
        xp: pd.DataFrame
            A new x after process
        """
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
    def drop_na(x: pd.DataFrame, axis=1, ratio=None) -> pd.DataFrame:
        """
        Drop na rows or columns that contains null values

        Parameters
        ----------
        x: pd.DataFrame
            A dataframe
        axis: int, default: 1
            if 1, drop columns; if 0 drop rows. Just supported 0 or 1.
        ratio: float, default: None
            drop rows/columns when the percentage of null values in rows/columns is grater than
            ratio.

        Returns
        -------
        xp: pd.DataFrame
            A new x after process
        """
        if ratio is None:
            ratio = 0
        thresh = round(x.shape[1 - axis] * (1 - ratio), 0)
        xp = x.dropna(axis=axis, thresh=thresh)
        if axis == 1:
            dlt = set(x.columns) - set(xp.columns)
        else:
            dlt = set(x.index) - set(xp.index)
        print(f"===== Dropping axis-{axis} the percentage of null values is > {ratio}: \n{dlt}")
        return xp
