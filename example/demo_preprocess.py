# coding - utf-8

import pandas as pd
from dmapp.data_preprocess import preprocessor
from dmapp.data_preprocess import DataPreprocessor


if __name__ == '__main__':
    x = pd.read_csv('../datasets/iris.csv')
    dp = DataPreprocessor(drop_na_features=True, drop_na_samples=True)
    xp = dp.execute(x)
    xp_numerics = preprocessor.get_numeric_features(xp)
    xp_objects = preprocessor.get_object_features(xp)
    print('complete')
