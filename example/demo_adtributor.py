# coding = utf-8

import numpy as np
import pandas as pd

from dmapp.rca.adtributor.adtributor import Adtributor


if __name__ == '__main__':
    np.random.seed(0)
    data_center = ['X', 'Y'] * 50
    ader = ['A1', 'A2', 'A3', 'A4'] * 25
    device_type = ['PC', 'Mobile', 'Table'] * 33 + ['PC']
    revenue = np.random.randint(5, 15, size=100)
    x = pd.DataFrame({'Data_Center': data_center,
                      'Advertizer': ader,
                      'Device_Type': device_type,
                      'Revenue': revenue})
    x['Forecasted_Revenue'] = x['Revenue'].rolling(window=10, min_periods=5, center=True).mean()
    yc = x[x['Forecasted_Revenue'] - x['Revenue'] > 3]
    dim_list = []
    for dim in yc.columns[:-2]:
        d_pivot = pd.pivot_table(yc, index=dim, values=['Revenue', 'Forecasted_Revenue'],
                                 aggfunc=np.mean)
        d_pivot.loc['All', :] = d_pivot.sum(axis=0)
        dim_list.append(d_pivot)
    out = Adtributor().root_cause_identification(dim_list)
    print(out)
