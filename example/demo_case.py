# coding - utf-8
import pandas as pd
from parrot.data_preprocess import preprocessor
from parrot.data_preprocess import DataPreprocessor


if __name__ == '__main__':
    x = pd.read_csv('../data/test.csv')
    dp = DataPreprocessor(x)
    dp.overview()
    xp = dp.quick_preprocess(drop_na_features=True, drop_na_samples=True)
    xp_numerics = preprocessor.get_numeric_features(xp)
    xp_objects = preprocessor.get_object_features(xp)
    print('complete')
