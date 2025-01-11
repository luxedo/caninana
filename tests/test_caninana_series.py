# SPDX-FileCopyrightText: 2025-present Luiz Eduardo Amaral <luizamaral306@gmail.com>
#
# SPDX-License-Identifier: MIT

import statistics
from types import MappingProxyType

import pytest

import lontras as lt

example_dict = MappingProxyType({"a": 1, "b": 2, "c": 3})
example_keys = tuple(example_dict.keys())
example_values = tuple(example_dict.values())
example_dict_a = MappingProxyType({"a": 1, "b": 2, "c": 3})
example_dict_b = MappingProxyType({"a": 4, "b": 5, "c": 6})
example_dict_no_keys = MappingProxyType({0: 1, 1: 2, 2: 3})
example_scalar = 3


class TestInit:
    def test_init_0(self):
        s = lt.Series()
        assert dict(s) == {}

    def test_init_1(self):
        s = lt.Series(example_dict.values(), keys=example_dict.keys())
        assert dict(s) == example_dict

    def test_init_2(self):
        s = lt.Series(example_dict)
        assert dict(s) == example_dict

    def test_init_3(self):
        s = lt.Series(tuple(example_dict.values()))
        assert dict(s) == example_dict_no_keys

    def test_init_4(self):
        s = lt.Series(0)
        assert s == {0: 0}

    def test_init_5(self):
        s = lt.Series(example_dict, keys=example_dict.keys())
        assert dict(s) == example_dict

    def test_init_error_0(self):
        with pytest.raises(ValueError, match="Length of data and keys must match"):
            lt.Series([0, 1, 2], [0, 1])

    def test_init_error_1(self):
        with pytest.raises(ValueError, match="Unexpected data type:"):
            lt.Series(int)

    def test_name(self):
        name = "snake"
        s = lt.Series(0, name=name)
        assert s.name == name

    def test_rename(self):
        name = "snake"
        s = lt.Series(0, name=name)
        new_name = "cobra"
        s.rename(new_name)
        assert s.name == new_name

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


class TestAccessors:
    def test_loc_getitem_scalar(self):
        s = lt.Series(example_dict)
        key = "a"
        assert s.loc[key] == example_dict[key]
        key = "b"
        assert s.loc[key] == example_dict[key]
        key = "c"
        assert s.loc[key] == example_dict[key]

    def test_loc_setitem_scalar(self):
        s = lt.Series(example_dict)
        key = "a"
        value = 4
        s.loc[key] = value
        assert s.loc[key] == value

    def test_loc_getitem_list(self):
        s = lt.Series(example_dict)
        keys = ["a", "b"]
        example_slice = {k: v for k, v in example_dict.items() if k in keys}
        assert (s.loc[keys] == example_slice).all()

    def test_loc_setitem_list(self):
        s = lt.Series(example_dict)
        keys = ["a", "b"]
        value = 4
        s.loc[keys] = value
        assert (s.loc[keys] == value).all()

    def test_iloc_getitem_scalar(self):
        s = lt.Series(example_dict)
        index = 0
        assert s.iloc[index] == example_values[index]
        index = 1
        assert s.iloc[index] == example_values[index]
        index = 2
        assert s.iloc[index] == example_values[index]

    def test_iloc_setitem_scalar(self):
        s = lt.Series(example_dict)
        index = 0
        value = 4
        s.iloc[index] = value
        assert s.iloc[index] == value

    def test_iloc_getitem_slice(self):
        s = lt.Series(example_dict)
        indexes_slice = slice(0, 2, 1)
        indexes = list(range(indexes_slice.start, indexes_slice.stop, indexes_slice.step))
        example_slice = {k: v for i, (k, v) in enumerate(example_dict.items()) if i in indexes}
        assert (s.iloc[indexes_slice] == example_slice).all()

    def test_iloc_setitem_slice(self):
        s = lt.Series(example_dict)
        indexes_slice = slice(0, 2, 1)
        value = 4
        s.iloc[indexes_slice] = value
        assert (s.iloc[indexes_slice] == value).all()

    def test_iloc_getitem_list(self):
        s = lt.Series(example_dict)
        indexes = [0, 1]
        example_slice = {k: v for i, (k, v) in enumerate(example_dict.items()) if i in indexes}
        assert (s.iloc[indexes] == example_slice).all()

    def test_iloc_setitem_list(self):
        s = lt.Series(example_dict)
        indexes = [0, 1]
        value = 4
        s.iloc[indexes] = value
        assert (s.iloc[indexes] == value).all()

    def test_head(self):
        s = lt.Series(example_dict)
        n = 2
        assert len(s.head(n)) == n
        assert (s.head(n) == example_values[:n]).all()

    def test_tail(self):
        s = lt.Series(example_dict)
        n = 2
        assert len(s.tail(n)) == n
        assert (s.tail(n) == example_values[-n:]).all()


class TestMapReduce:
    def test_map(self):
        s = lt.Series(example_dict)
        assert s.map(lambda x: x**2) == {k: v**2 for k, v in example_dict.items()}

    def test_reduce(self):
        s = lt.Series(example_dict)
        assert s.reduce(lambda acc, cur: acc + cur[0], "") == "".join(example_dict.keys())
        s = lt.Series()
        assert s.reduce(lambda *_: 0, 0) == 0

    def test_max(self):
        s = lt.Series(example_dict)
        assert s.max() == 3

    def test_min(self):
        s = lt.Series(example_dict)
        assert s.min() == 1

    def test_sum(self):
        s = lt.Series(example_dict)
        assert s.sum() == 6

    def test_all(self):
        s = lt.Series(example_dict)
        assert s.all()
        s = lt.Series([0, 1, 2])
        assert not s.all()

    def test_any(self):
        s = lt.Series(example_dict)
        assert s.any()
        s = lt.Series([0, 1, 2])
        assert s.any()
        s = lt.Series([0])
        assert not s.any()

    def test_argmax(self):
        s = lt.Series(example_dict)
        assert s.argmax() == "c"

    def test_argmin(self):
        s = lt.Series(example_dict)
        assert s.argmin() == "a"


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
            "median_low",
            "median_high",
            "median_grouped",
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

    @pytest.mark.parametrize(
        "func",
        [
            "covariance",
            "correlation",
            "linear_regression",
        ],
    )
    def test_statistics_other(self, func):
        sa = lt.Series(example_dict_a)
        sb = lt.Series(example_dict_b)
        assert getattr(sa, func)(sb) == getattr(statistics, func)(example_dict_a.values(), example_dict_b.values())


class TestExports:
    def test_to_list(self):
        s = lt.Series(example_dict)
        assert s.to_list() == list(example_values)

    def test_to_dict(self):
        s = lt.Series(example_dict)
        assert s.to_dict() == example_dict


class TestComparisons:
    def test_lt_ge(self):
        sa = lt.Series([0, 1])
        sb = lt.Series([1, 2])
        assert (sa < sb).all()
        assert not (sa >= sb).all()

    def test_le_gt(self):
        sa = lt.Series([0, 1])
        sb = lt.Series([0, 2])
        assert (sa <= sb).all()
        assert not (sa > sb).all()

    def test_eq(self):
        sa = lt.Series([0, 1])
        sb = lt.Series([0, 1])
        assert (sa == sb).all()
        sa = lt.Series([0, 1])
        sb = lt.Series([0, 2])
        assert not (sa == sb).all()

    def test_ne(self):
        sa = lt.Series([0, 1])
        sb = lt.Series([1, 2])
        assert (sa != sb).all()
        sa = lt.Series([0, 1])
        sb = lt.Series([0, 1])
        assert not (sa != sb).all()


class TestOperators:
    @pytest.mark.parametrize(
        "op",
        [
            "__add__",
            "__sub__",
            "__mul__",
            # "__matmul__",
            "__truediv__",
            "__floordiv__",
            "__mod__",
            "__divmod__",
            "__pow__",
            "__lshift__",
            "__rshift__",
            "__and__",
            "__xor__",
            "__or__",
            "__radd__",
            "__rsub__",
            "__rmul__",
            # "__matmul__",
            "__rtruediv__",
            "__rfloordiv__",
            "__rmod__",
            "__rdivmod__",
            "__rpow__",
            "__rlshift__",
            "__rrshift__",
            "__rand__",
            "__rxor__",
            "__ror__",
        ],
    )
    def test_op(self, op):
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
        ("iop", "op"),
        [
            ("__iadd__", "__add__"),
            ("__isub__", "__sub__"),
            ("__imul__", "__mul__"),
            # ("__imatmul__","__matmul__"),
            ("__itruediv__", "__truediv__"),
            ("__ifloordiv__", "__floordiv__"),
            ("__imod__", "__mod__"),
            ("__ipow__", "__pow__"),
            ("__ilshift__", "__lshift__"),
            ("__irshift__", "__rshift__"),
            ("__iand__", "__and__"),
            ("__ixor__", "__xor__"),
            ("__ior__", "__or__"),
        ],
    )
    def test_op_inplace(self, iop, op):
        sa = lt.Series(example_dict_a)
        sb = lt.Series(example_dict_b)
        getattr(sa, iop)(sb)
        assert sa == {k: getattr(v, op)(example_dict_b[k]) for k, v in example_dict_a.items()}


class TestUnaryOperators:
    def test_neg(self):
        values = [-1, 0, 1]
        s = lt.Series(values)
        assert dict(-s) == {i: -v for i, v in enumerate(values)}

    def test_pos(self):
        values = [-1, 0, 1]
        s = lt.Series(values)
        assert dict(+s) == {i: +v for i, v in enumerate(values)}

    def test_abs(self):
        values = [-1, 0, 1]
        s = lt.Series(values)
        assert dict(abs(s)) == {i: abs(v) for i, v in enumerate(values)}

    def test_invert(self):
        values = [-1, 0, 1]
        s = lt.Series(values)
        assert dict(~s) == {i: ~v for i, v in enumerate(values)}
