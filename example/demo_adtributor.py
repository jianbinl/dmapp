# coding = utf-8

import numpy as np
import pandas as pd

from dmapp.fault_localization.adtributor.adtributor import RootCauseLocation


if __name__ == '__main__':
    df = pd.read_csv('../datasets/1450653900.csv')
    print(df.head())
    dimensions = sorted(list(set(df.columns) - {'real', 'predict'}))
    res = RootCauseLocation.by_adtributor_recursive(df, dimensions=dimensions, k=3, revised=True)
    print(res)
