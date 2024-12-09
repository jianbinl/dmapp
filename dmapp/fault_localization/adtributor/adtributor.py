# coding=utf-8

import numpy as np
import pandas as pd


def create_cubes(df, dimensions, derived):
    df = RootCauseLocation.check_input(df, derived)
    cubes = []
    for each_dim in dimensions:
        if derived:
            elements = pd.pivot_table(df, values=['real_x', 'real_y', 'predict_x', 'predict_y'],
                                      index=each_dim, aggfunc='sum')
        else:
            elements = pd.pivot_table(df, values=['real', 'predict'], index=each_dim, aggfunc='sum')
        elements.reset_index(drop=False, inplace=True)
        cubes.append(elements)
    cubes = pd.concat(cubes, axis=0, ignore_index=True)
    return cubes


def js_divergence(p, q):
    p_part = np.nan_to_num(p * np.log(2 * p / (p + q)))
    q_part = np.nan_to_num(q * np.log(2 * q / (p + q)))
    js = 0.5 * (p_part + q_part)
    return js


def compute_surprise(cubes, derived, stack_dims=1):
    """

    Parameters
    ----------
    cubes
    derived
    stack_dims: int, default: 1
        denotes how many dims in cubes have been merged.

    Returns
    -------

    """
    def surprise(real_predict_df):
        """real_predict_df must contain two columns: 'real' and 'predict'."""
        f = real_predict_df['predict'].sum() / stack_dims
        a = real_predict_df['real'].sum() / stack_dims
        with np.errstate(divide='ignore'):
            p = real_predict_df['predict'] / f
            q = real_predict_df['real'] / a
            s = js_divergence(p, q)
        return s

    if derived:
        s = surprise(cubes[['real_a', 'predict_a']]) + surprise(cubes[['real_b', 'predict_b']])
    else:
        s = surprise(cubes)
    cubes['surprise'] = s
    return cubes


def compute_explanatory_power(cubes, derived):
    if derived:
        f1 = cubes['predict_x'].sum()
        f2 = cubes['predict_y'].sum()
        ep = ((cubes['real_x'] - cubes['predict_x']) * f2 -
              (cubes['real_y'] - cubes['predict_y']) * f1) / \
             (f2 * (f2 + (cubes['real_y'] - cubes['predict_y'])))
        ep = ep / ep.sum()
    else:
        a = cubes['real'].sum()
        f = cubes['predict'].sum()
        ep = (cubes['real'] - cubes['predict']) / (a - f)
    cubes['explanatory_power'] = ep
    return cubes


def compute_s_and_ep(df, dimensions, derived=False):
    assert dimensions, f"dimensions can't be empty set."
    cubes = create_cubes(df, dimensions, derived)
    cubes = compute_surprise(cubes, derived, stack_dims=len(dimensions))
    cubes = compute_explanatory_power(cubes, derived)
    return cubes


class RootCauseLocation:

    def __init__(self):
        ...

    @staticmethod
    def check_input(df, derived=False):
        if derived:
            assert {'real_x', 'real_y', 'predict_x', 'predict_y'} - set(df.columns), \
                (f"Please set real and predict column name of fundamental measures to "
                 f"['real_x', 'real_y', 'predict_x', 'predict_y'], "
                 f"`_x` is suffix of numerator and `_y` is suffix of denominator.")
        else:
            assert not {'real', 'predict'} - set(df.columns), \
                (f"Please set real and predict column name of derived measures to 'real' and "
                 f"'predict'.")
        return df

    @classmethod
    def by_adtributor(cls, df, dimensions, teep=0.01, tep=0.1, k=None, derived=False):
        """

        Parameters
        ----------
        df: pd.DataFrame
            if derived=True, it must contains 4 columns: real_x, predict_x, real_y, predict_y;
            if derives=False, it must contains 2 columns: real, predict.
        dimensions: Iterable
            List of str, Category features use for location root cause
        teep: float, default: 0.1
        tep: float, default: 0.1
            Efficiently when revised=False
        k: int, default: 3
            Return top k root cause
        derived: bool, default: False
            if measure dimension is derived measure.

        Returns
        -------
        out: Iterable
            Top k root cause
        """
        cubes = compute_s_and_ep(df, dimensions, derived)
        explanatory_set = []
        for dim in dimensions:
            sub_cube = cubes[~cubes[dim].isnull()]
            sub_cube.set_index(dim, drop=True, inplace=True)
            sub_cube_sorted = sub_cube.sort_values('surprise', ascending=False)
            candidate_set = sub_cube_sorted.loc[sub_cube_sorted['explanatory_power'] > teep,
                                                'explanatory_power'].cumsum()
            if np.any(candidate_set > tep):
                idx = (candidate_set > tep).idxmax()
                candidate = {
                    'dimension': sub_cube.index.name,
                    'elements': candidate_set[:idx].index.to_list(),
                    'surprise': sub_cube_sorted.loc[:idx, 'surprise'].sum(),
                    'explanatory_power': candidate_set[idx]
                }
                explanatory_set.append(candidate)
        explanatory_set = sorted(explanatory_set, key=lambda x: x['surprise'], reverse=True)
        if k is None:
            k = len(explanatory_set)
        return explanatory_set[:k]

    @classmethod
    def by_adtributor_revised(cls, df, dimensions, teep=0.1, interval=0.1, k=None, derived=False):
        cubes = compute_s_and_ep(df, dimensions, derived)
        explanatory_set = []
        for dim in dimensions:
            sub_cube = cubes[~cubes[dim].isnull()]
            sub_cube.set_index(dim, drop=True, inplace=True)
            outside = np.logical_or((sub_cube['predict'] > sub_cube['real'] * (1 + interval)),
                                    (sub_cube['predict'] < sub_cube['real'] * (1 - interval)))
            candidate_set = sub_cube.loc[(sub_cube['explanatory_power'] > teep) & outside,
                                         'explanatory_power']
            if 0 < candidate_set.shape[0] < sub_cube.shape[0]:
                candidate = {
                    'dimension': sub_cube.index.name,
                    'elements': candidate_set.index.to_list(),
                    'surprise': sub_cube.loc[:, 'surprise'].sum(),
                    'explanatory_power': candidate_set.sum()
                }
                explanatory_set.append(candidate)
        explanatory_set = sorted(explanatory_set, key=lambda x: x['surprise'], reverse=True)
        if k is None:
            k = len(explanatory_set)
        return explanatory_set[:k]

    @classmethod
    def by_adtributor_recursive(cls, df, dimensions, teep=0.1, tep=0.1, interval=0.1,
                                k=None, derived=False, revised=False):
        """

        References
        ----------
        [1]: https://odr.chalmers.se/server/api/core/bitstreams/1641e4bf-edec-4fe3-b1ed-0c281d538824/content

        Parameters
        ----------
        df: pd.DataFrame
            if derived=True, it must contains 4 columns: real_x, predict_x, real_y, predict_y;
            if derives=False, it must contains 2 columns: real, predict.
        dimensions: Iterable
            List of str, Category features use for location root cause
        teep: float, default: 0.1
        tep: float, default: 0.1
            Efficiently when revised=False
        interval: float, default: 0.1
            (0, 1), denotes predict interval in [(1 - interval) * predict, (1 + interval) * predict]
        k: int, default: 3
            Return top k root cause
        derived: bool, default: False
            if measure dimension is derived measure.
        revised: bool, default: False
            if use revised adtributor algorithm.

        Returns
        -------
        out: Iterable
            multiple combinations of dimensions for top k root cause

        Examples
        --------
        >>> df = pd.read_csv('../../../datasets/1450653900.csv')
        >>> df.head()
           real  predict   a    b   c   d
        0   0.0      0.0  a5  b17  c5  d1
        1   0.0      0.0  a5  b17  c5  d4
        2   0.0      0.0  a5  b17  c5  d9
        3   0.0      0.0  a5  b17  c5  d5
        4   0.0      0.0  a5  b17  c5  d3
        >>> dimensions = sorted(list(set(df.columns) - {'real', 'predict'}))
        >>> res = RootCauseLocation.by_adtributor_recursive(df, dimensions=dimensions, k=3,
        revised=True)
        >>> print(res)
        """
        def drop_duplicates(explanatory_set):
            d = dict()
            for c in explanatory_set:
                es = ''.join(np.array(c['elements']).flatten())
                if es not in d.keys():
                    d[es] = c
            return list(d.values())

        if revised:
            explanatory_set = cls.by_adtributor_revised(df, dimensions, teep, interval, k, derived)
        else:
            explanatory_set = cls.by_adtributor(df, dimensions, teep, tep, k, derived)
        new_explanatory_set = []
        for candidate_set in explanatory_set:
            candidate_set['cuboid'] = [candidate_set['dimension']]
            candidate_set['elements'] = [[e] for e in candidate_set['elements']]
            new_dimensions = set(dimensions) - {candidate_set['dimension']}
            if not new_dimensions:
                continue
            new_candidate_set = []
            for current_can in candidate_set['elements']:
                df_c = df[df[candidate_set['dimension']] == current_can[0]].copy()
                c_explanatory_set = cls.by_adtributor_recursive(
                    df=df_c, dimensions=new_dimensions, teep=teep, tep=tep, interval=interval, k=k,
                    derived=derived, revised=revised
                )
                if len(c_explanatory_set) == 0:
                    new_candidate_set = []
                    break
                for each_c in c_explanatory_set:
                    each_c['elements'] = [current_can + e for e in each_c['elements']]
                    each_c['explanatory_power'] = each_c['explanatory_power'] * candidate_set['explanatory_power']
                    each_c['cuboid'] = candidate_set['cuboid'] + each_c['cuboid']
                new_candidate_set.extend(c_explanatory_set)
            if new_candidate_set:
                new_explanatory_set.extend(new_candidate_set)
            else:
                new_explanatory_set.append(candidate_set)
        new_explanatory_set = drop_duplicates(new_explanatory_set)
        return new_explanatory_set
