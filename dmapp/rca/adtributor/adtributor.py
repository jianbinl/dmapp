# coding = utf-8
import numpy as np
import pandas as pd


def js_divergence(p, q):
    """
    Calculate JS Divergence

    Parameters
    ----------
    p: array-like
        probability distribution
    q: array-like
        probability distribution

    Returns
    -------
    js: float
        JS Divergence
    """

    js = 0.5 * (p * np.log2(2 * p / (p + q)) + q * np.log2(2 * q / (p + q)))
    return js


class Adtributor(object):
    """
    Root cause analysis

    References
    ---------
    [1]: https://www.usenix.org/system/files/conference/nsdi14/nsdi14-paper-bhagwan.pdf
    [2]: https://arxiv.org/pdf/2205.10004

    Parameters
    ----------

    Examples
    --------
    >>> np.random.seed(0)
    >>> data_center = ['X', 'Y'] * 50
    >>> ader = ['A1', 'A2', 'A3', 'A4'] * 25
    >>> device_type = ['PC', 'Mobile', 'Table'] * 33 + ['PC']
    >>> revenue = np.random.randint(5, 15, size=100)
    >>> x = pd.DataFrame({'Data_Center': data_center,
    ...                   'Advertizer': ader,
    ...                   'Device_Type': device_type,
    ...                   'Revenue': revenue})
    >>> x['Forecasted_Revenue'] = x['Revenue'].rolling(window=10, min_periods=5, center=True).mean()
    >>> yc = x[x['Forecasted_Revenue'] - x['Revenue'] > 3]
    >>> dim_list = []
    >>> for dim in yc.columns[:-2]:
    ...     d_pivot = pd.pivot_table(yc, index=dim, values=['Revenue', 'Forecasted_Revenue'],
    ...                              aggfunc=np.sum)
    ...     d_pivot.loc['ALL', :] = d_pivot.sum(axis=0)
    ...     dim_list.append(d_pivot)
    >>> out = Adtributor().root_cause_identification(dim_list)
    >>> print(out)
    """

    @staticmethod
    def check_input(inputs):
        for input_ in inputs:
            assert isinstance(input_, pd.DataFrame), 'all input in inputs must be pd.DataFrame.'
            assert input_.index.name is not None, \
                'The index name of all input in inputs must not None.'
            assert input_.shape[1] == 2, 'Each dimension pivot must has only two columns.'
            assert np.all(input_.iloc[:-1, :].sum(axis=0) == input_.iloc[-1, :]), \
                'The last row must be summation of all above rows.'
        return inputs

    @staticmethod
    def cal_surprise(df_list):
        for each_df in df_list:
            p = each_df.iloc[:-1, 0] / each_df.iloc[-1, 0]
            q = each_df.iloc[:-1, 1] / each_df.iloc[-1, 1]
            js = js_divergence(p, q)
            js = np.append(js, np.sum(js))
            each_df['S'] = js
        return df_list

    def root_cause_identification(self, df_list, Teep=0.001, Tep=0.01, top_k=None):
        """
        Root cause identification

        Parameters
        ----------
        df_list: list
            Each element is a pivot for each dimension
        Teep: float
            Based on Occam's razor, exclude element which EP is smaller than Teep.
        Tep: float
            Based on Occam's razor, when EP is larger than Tep, ignore the rest of elements.
        top_k: int, None
            Return top_k root cause

        Returns
        -------
        final: pd.DataFrame
            top_k root cause
        """

        df_list = self.check_input(df_list)
        df_list_s = self.cal_surprise(df_list)
        explanatory_set = {'dimension': [],
                           'elements': [],
                           'S': [],
                           'EP': []}
        for dim_df in df_list_s:
            sorted_dim_df = dim_df.iloc[:-1, :].sort_values('S', ascending=False)
            candidate = []
            explains = 0
            surprise = 0
            for element in sorted_dim_df.index:
                col = dim_df.columns
                ep = ((dim_df.loc[element, col[0]] - dim_df.loc[element, col[1]])
                      / (dim_df.iloc[-1, 0] - dim_df.iloc[-1, 1]))
                if ep > Teep:
                    explains += ep
                    surprise += dim_df.loc[element, 'S']
                    candidate.append(element)
                if explains > Tep:
                    explanatory_set['dimension'].append(dim_df.index.name)
                    explanatory_set['elements'].append(candidate)
                    explanatory_set['S'].append(surprise)
                    explanatory_set['EP'].append(explains)
                    break
        final = pd.DataFrame(explanatory_set).sort_values('S', ascending=False)
        if top_k is None:
            top_k = final.shape[0]
        return final.iloc[:top_k, :]
