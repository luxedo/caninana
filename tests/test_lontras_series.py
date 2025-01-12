# SPDX-FileCopyrightText: 2025-present Luiz Eduardo Amaral <luizamaral306@gmail.com>
#
# SPDX-License-Identifier: MIT

import re
import statistics
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

import lontras as lt

example_dict = MappingProxyType({"a": 1, "b": 2, "c": 3})
example_index = tuple(example_dict.keys())
example_values = tuple(example_dict.values())
example_dict_a = MappingProxyType({"a": 1, "b": 2, "c": 3})
example_dict_b = MappingProxyType({"a": 4, "b": 5, "c": 6})
example_dict_no_keys = MappingProxyType({0: 1, 1: 2, 2: 3})
example_scalar = 3


class TestInit:
    def test_init_0(self):
        s = lt.Series()
        ps = pd.Series()
        assert (s == ps).all()

    def test_init_1(self):
        s = lt.Series(example_dict.values(), index=example_index)
        ps = pd.Series(example_dict.values(), index=example_index)
        assert (s == ps).all()

    def test_init_2(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        assert (s == ps).all()

    def test_init_3(self):
        s = lt.Series(tuple(example_dict.values()))
        ps = pd.Series(tuple(example_dict.values()))
        assert (s == ps).all()

    def test_init_4(self):
        s = lt.Series(0)
        ps = pd.Series(0)
        assert s == {0: 0}
        assert (s == ps).all()

    def test_init_5(self):
        s = lt.Series(example_dict, index=example_index)
        ps = pd.Series(example_dict, index=example_index)
        assert (s == ps).all()

    def test_init_error_0(self):
        try:
            pd.Series([0, 1, 2], [0, 1])
        except ValueError as e:
            # with pytest.raises(ValueError, match="Length of data and index must match"):
            with pytest.raises(ValueError, match=re.escape(str(e))):
                lt.Series([0, 1, 2], [0, 1])

    def test_init_error_1(self):
        with pytest.raises(ValueError, match="Unexpected data type:"):
            lt.Series(int)

    def test_name(self):
        name = "snake"
        s = lt.Series(0, name=name)
        ps = pd.Series(0, name=name)
        assert s.name == ps.name

    def test_rename(self):
        name = "snake"
        s = lt.Series(0, name=name)
        ps = pd.Series(0, name=name)
        new_name = "cobra"
        s.rename(new_name)  # Should not mutate
        ps.rename(new_name)
        assert s.name == ps.name
        assert s.rename(new_name) == ps.rename(new_name)

    def test_shallow_copy(self):
        s = lt.Series([[123]])
        t = s.copy(deep=False)
        s.iloc[0][0] = [456]
        assert (t == s).all()

    def test_deepcopy(self):
        s = lt.Series([[123]])
        t = s.copy()
        s.iloc[0][0] = [456]
        assert (t != s).all()

    def test_index_getter(self):
        s = lt.Series(example_dict)
        assert s.index == list(example_index)

    def test_index_setter(self):
        s = lt.Series(example_dict)
        s.index = list(reversed(example_index))
        assert s.index == list(reversed(example_index))

    def test_index_setter_error(self):
        s = lt.Series(example_dict)
        with pytest.raises(ValueError, match="Length mismatch"):
            s.index = [*list(example_index), "more_indexes"]


class TestAccessors:
    def test_getitem_0(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        key = "a"
        assert s[key] == ps[key]
        key = "b"
        assert s[key] == ps[key]
        key = "c"
        assert s[key] == ps[key]

    def test_getitem_1(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        indexes = ["a", "c"]
        assert (s[indexes] == ps[indexes]).all()

    def test_getitem_2(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        indexes = slice(0, 2)
        assert (s[indexes] == ps[indexes]).all()

    def test_getitem_3(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        val = 2
        mask_s = s > val
        mask_ps = ps > val
        assert (s[mask_s] == ps[mask_ps]).all()

    def test_getitem_4(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        indexes = lt.Series(example_index[:2])
        assert (s[indexes] == ps[indexes]).all()

    def test_loc_getitem_scalar(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        key = "a"
        assert s.loc[key] == ps.loc[key]
        key = "b"
        assert s.loc[key] == ps.loc[key]
        key = "c"
        assert s.loc[key] == ps.loc[key]

    def test_loc_setitem_scalar(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        key = "a"
        value = 4
        s.loc[key] = value
        ps.loc[key] = value
        assert s.loc[key] == ps[key]

    def test_loc_getitem_list(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        keys = ["a", "b"]
        assert (s.loc[keys] == ps.loc[keys]).all()

    def test_loc_setitem_list(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        keys = ["a", "b"]
        value = 4
        s.loc[keys] = value
        ps.loc[keys] = value
        assert (s.loc[keys] == ps.loc[keys]).all()

    def test_loc_get_not_hashable_key(self):
        s = lt.Series(example_dict)
        with pytest.raises(TypeError, match="Cannot index"):
            s.loc[{1, 2, 3}]

    def test_loc_set_not_hashable_key(self):
        s = lt.Series(example_dict)
        with pytest.raises(TypeError, match="Cannot index"):
            s.loc[{1, 2, 3}] = "no!"

    def test_iloc_getitem_scalar(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        index = 0
        assert s.iloc[index] == ps.iloc[index]
        index = 1
        assert s.iloc[index] == ps.iloc[index]
        index = 2
        assert s.iloc[index] == ps.iloc[index]

    def test_iloc_setitem_scalar(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        index = 0
        value = 4
        s.iloc[index] = value
        ps.iloc[index] = value
        assert s.iloc[index] == ps.iloc[index]

    def test_iloc_getitem_slice(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        indexes = slice(0, 2, 1)
        assert (s.iloc[indexes] == ps.iloc[indexes]).all()

    def test_iloc_setitem_slice(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        indexes = slice(0, 2, 1)
        value = 4
        s.iloc[indexes] = value
        ps.iloc[indexes] = value
        assert (s.iloc[indexes] == ps.iloc[indexes]).all()

    def test_iloc_getitem_list(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        indexes = [0, 1]
        assert (s.iloc[indexes] == ps.iloc[indexes]).all()

    def test_iloc_setitem_list(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        indexes = [0, 1]
        value = 4
        s.iloc[indexes] = value
        ps.iloc[indexes] = value
        assert (s.iloc[indexes] == ps.iloc[indexes]).all()

    def test_iloc_get_error(self):
        s = lt.Series(example_dict)
        with pytest.raises(TypeError, match="Cannot index"):
            s.iloc[{1, 2, 3}]

    def test_loc_set_error(self):
        s = lt.Series(example_dict)
        with pytest.raises(TypeError, match="Cannot index"):
            s.iloc[{1, 2, 3}] = "no!"

    def test_head(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        n = 2
        assert len(s.head(n)) == n
        assert (s.head(n) == example_values[:n]).all()
        assert (s.head(n) == ps.head(n)).all()

    def test_tail(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        n = 2
        assert len(s.tail(n)) == n
        assert (s.tail(n) == example_values[-n:]).all()
        assert (s.tail(n) == ps.tail(n)).all()

    def test_find(self):
        s = lt.Series(example_dict)
        key = "a"
        assert s.find(example_dict[key]) == key

    def test_find_not_found(self):
        s = lt.Series(example_dict)
        assert s.find("value not found") is None

    def test_ifind(self):
        s = lt.Series(example_dict)
        index = 0
        key = list(example_index)[index]
        assert s.ifind(example_dict[key]) == index

    def test_ifind_not_found(self):
        s = lt.Series(example_dict)
        assert s.ifind("value not found") is None


class TestMapReduce:
    def test_map(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        assert (s.map(lambda x: x**2) == ps.map(lambda x: x**2)).all()

    def test_reduce(self):
        s = lt.Series(example_dict)
        assert s.reduce(lambda acc, cur: acc + cur[0], "") == "".join(example_index)
        s = lt.Series()
        assert s.reduce(lambda *_: 0, 0) == 0

    @pytest.mark.parametrize(
        "func",
        [
            "max",
            "min",
            "sum",
            "all",
            "any",
            "argmax",
            "argmin",
            "idxmax",
            "idxmin",
        ],
    )
    def test_aggregations(self, func):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        assert getattr(s, func)() == getattr(ps, func)()

    def test_all(self):
        s = lt.Series([0, 1, 2])
        ps = pd.Series([0, 1, 2])
        assert s.all() == ps.all()

    def test_any(self):
        s = lt.Series([0, 1, 2])
        ps = pd.Series([0, 1, 2])
        assert s.any() == ps.any()
        s = lt.Series([0])
        ps = pd.Series([0])
        assert s.any() == ps.any()

    def test_astype(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        assert (s.astype(str) == ps.astype(str)).all()

    @pytest.mark.parametrize(
        "func",
        [
            "argmax",
            "argmin",
            "idxmax",
            "idxmin",
        ],
    )
    def test_arg_idx_errors(self, func):
        s = lt.Series()
        with pytest.raises(ValueError, match="empty sequence"):
            getattr(s, func)()


class TestStatistics:
    @pytest.mark.parametrize(
        "func",
        [
            "mean",
            "fmean",
            "geometric_mean",
            "harmonic_mean",
            # "kde",
            # "kde_random",
            "median",
            "mode",
            "multimode",
            "quantiles",
            "pstdev",
            "pvariance",
            "stdev",
            "variance",
        ],
    )
    def test_statistics(self, func):
        s = lt.Series(example_dict)
        assert getattr(s, func)() == getattr(statistics, func)(example_dict.values())

    # @pytest.mark.parametrize(
    #     "func",
    #     [
    #         "covariance",
    #         "correlation",
    #         "linear_regression",
    #     ],
    # )
    # def test_statistics_other(self, func):
    #     sa = lt.Series(example_dict_a)
    #     sb = lt.Series(example_dict_b)
    #     assert getattr(sa, func)(sb) == getattr(statistics, func)(example_dict_a.values(), example_dict_b.values())


class TestExports:
    def test_to_list(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        assert s.to_list() == ps.to_list()

    def test_to_dict(self):
        s = lt.Series(example_dict)
        ps = pd.Series(example_dict)
        assert s.to_dict() == ps.to_dict()


class TestComparisons:
    def test_lt_ge(self):
        sa = lt.Series([0, 1])
        sb = lt.Series([1, 2])
        psa = pd.Series([0, 1])
        psb = pd.Series([1, 2])
        assert ((sa < sb) == (psa < psb)).all()
        assert ((sa >= sb) == (psa >= psb)).all()

    def test_le_gt(self):
        sa = lt.Series([0, 1])
        sb = lt.Series([0, 2])
        psa = pd.Series([0, 1])
        psb = pd.Series([0, 2])
        assert ((sa <= sb) == (psa <= psb)).all()
        assert ((sa > sb) == (psa > psb)).all()

    def test_eq(self):
        sa = lt.Series([0, 1])
        sb = lt.Series([0, 1])
        psa = pd.Series([0, 1])
        psb = pd.Series([0, 1])
        assert ((sa == sb) == (psa == psb)).all()
        sa = lt.Series([0, 1])
        sb = lt.Series([0, 2])
        psa = pd.Series([0, 1])
        psb = pd.Series([0, 2])
        assert ((sa == sb) == (psa == psb)).all()

    def test_ne(self):
        sa = lt.Series([0, 1])
        sb = lt.Series([1, 2])
        psa = pd.Series([0, 1])
        psb = pd.Series([1, 2])
        assert ((sa != sb) == (psa != psb)).all()
        sa = lt.Series([0, 1])
        sb = lt.Series([0, 1])
        psa = pd.Series([0, 1])
        psb = pd.Series([0, 1])
        assert ((sa != sb) == (psa != psb)).all()


class TestOperators:
    @pytest.mark.parametrize(
        "op",
        [
            "__add__",
            "__sub__",
            "__mul__",
            "__truediv__",
            "__floordiv__",
            "__mod__",
            "__pow__",
            "__radd__",
            "__rsub__",
            "__rmul__",
            "__rtruediv__",
            "__rfloordiv__",
            "__rmod__",
            "__rpow__",
        ],
    )
    def test_op(self, op):
        sa = lt.Series(example_dict_a)
        sb = lt.Series(example_dict_b)
        psa = pd.Series(example_dict_a)
        psb = pd.Series(example_dict_b)

        # Series
        assert (getattr(sa, op)(sb) == getattr(psa, op)(psb)).all()
        # Scalar
        assert (getattr(sa, op)(example_scalar) == getattr(psa, op)(example_scalar)).all()
        # Collection
        assert (getattr(sa, op)(example_values) == getattr(psa, op)(example_values)).all()

    def test_matmul(self):
        sa = lt.Series(example_dict_a)
        sb = lt.Series(example_dict_b)
        psa = pd.Series(example_dict_a)
        psb = pd.Series(example_dict_b)

        # Series
        assert (sa @ sb) == (psa @ psb)
        # Collection
        assert (sa @ example_values) == (psa @ example_values)

    @pytest.mark.parametrize(
        "op",
        [
            "__and__",
            "__xor__",
            "__or__",
            "__rand__",
            "__rxor__",
            "__ror__",
        ],
    )
    def test_bop(self, op):
        # Series
        sa = lt.Series(example_dict_a)
        sb = lt.Series(example_dict_b)
        psa = pd.Series(example_dict_a)
        psb = pd.Series(example_dict_b)
        assert (getattr(sa, op)(sb) == getattr(psa, op)(psb)).all()

        # Scalar
        sa = lt.Series(example_dict_a)
        assert (getattr(sa, op)(example_scalar) == getattr(psa, op)(example_scalar)).all()

        # Collection
        sa = lt.Series(example_dict_a)
        assert (getattr(sa, op)(example_values) == getattr(psa, op)(np.array(example_values))).all()
        # Pandads is deprecating logical ops for dtype-less sequqences (eg: list, tuple)

    @pytest.mark.parametrize(
        "op",
        [
            "__divmod__",
            "__lshift__",
            "__rshift__",
            "__rdivmod__",
            "__rlshift__",
            "__rrshift__",
        ],
    )
    def test_rop(self, op):
        # Series
        sa = lt.Series(example_dict_a)
        sb = lt.Series(example_dict_b)
        assert getattr(sa, op)(sb) == {k: getattr(v, op)(example_dict_b[k]) for k, v in example_dict_a.items()}

        # Scalar
        sa = lt.Series(example_dict_a)
        assert getattr(sa, op)(example_scalar) == {k: getattr(v, op)(example_scalar) for k, v in example_dict_a.items()}

        # Collection
        sa = lt.Series(example_dict_a)
        assert getattr(sa, op)(example_values) == {
            k: getattr(v, op)(example_values[i]) for i, (k, v) in enumerate(example_dict_a.items())
        }

    @pytest.mark.parametrize(
        "iop",
        [
            "__iadd__",
            "__isub__",
            "__imul__",
            # "__imatmul__",
            "__itruediv__",
            "__ifloordiv__",
            "__imod__",
            "__ipow__",
            "__iand__",
            "__ixor__",
            "__ior__",
        ],
    )
    def test_op_inplace(self, iop):
        sa = lt.Series(example_dict_a)
        sb = lt.Series(example_dict_b)
        psa = pd.Series(example_dict_a)
        psb = pd.Series(example_dict_b)
        getattr(sa, iop)(sb)
        getattr(psa, iop)(psb)
        assert (sa == psa).all()

    @pytest.mark.parametrize(
        ("iop", "op"),
        [
            ("__ilshift__", "__lshift__"),
            ("__irshift__", "__rshift__"),
        ],
    )
    def test_op_inplace_shift(self, iop, op):
        sa = lt.Series(example_dict_a)
        sb = lt.Series(example_dict_b)
        getattr(sa, iop)(sb)
        assert sa == {k: getattr(v, op)(example_dict_b[k]) for k, v in example_dict_a.items()}

    def test_different_length_op_error(self):
        sa = lt.Series(example_dict_a)
        sb = lt.Series(example_dict_b)[: len(example_dict_b) - 2]
        with pytest.raises(ValueError, match="Cannot operate"):
            sa + sb


class TestUnaryOperators:
    def test_neg(self):
        values = [-1, 0, 1]
        s = lt.Series(values)
        ps = pd.Series(values)
        assert (-s == -ps).all()

    def test_pos(self):
        values = [-1, 0, 1]
        s = lt.Series(values)
        ps = pd.Series(values)
        assert (+s == +ps).all()

    def test_abs(self):
        values = [-1, 0, 1]
        s = lt.Series(values)
        ps = pd.Series(values)
        assert (abs(s) == abs(ps)).all()

    def test_invert(self):
        values = [-1, 0, 1]
        s = lt.Series(values)
        ps = pd.Series(values)
        assert (~s == ~ps).all()
