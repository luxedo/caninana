# SPDX-FileCopyrightText: 2025-present Luiz Eduardo Amaral <luizamaral306@gmail.com>
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import copy
import statistics
from collections import UserDict
from collections.abc import Callable, Collection, Hashable, Mapping
from functools import reduce
from typing import Any, Literal, TypeAlias

Scalar: TypeAlias = int | float | complex | str | bool
Keys: TypeAlias = Collection[Hashable]


class LocIndexer:
    def __init__(self, series: Series):
        self.series = series

    def __getitem__(self, key: Hashable | list[Hashable]) -> Any:
        if isinstance(key, list):
            return Series({k: self.series[k] for k in key})
        if isinstance(key, Hashable):
            return self.series[key]
        msg = f"Cannot index with: {key=}"
        raise TypeError(msg)

    def __setitem__(self, key: Hashable | list[Hashable], value: Any) -> Any:
        if isinstance(key, list):
            for k in key:
                self.series[k] = value
            return
        if isinstance(key, Hashable):
            self.series[key] = value
            return
        msg = f"Cannot index with: {key=}"
        raise TypeError(msg)


class IlocIndexer:
    def __init__(self, series: Series):
        self.series = series
        self.keys = list(series.keys())

    def __getitem__(self, index: int | slice | list[int]) -> Any:
        if isinstance(index, int):
            return self.series[self.keys[index]]
        if isinstance(index, slice):
            keys = self.keys[index]
            return Series({k: self.series[k] for k in keys})
        if isinstance(index, list):
            keys = [self.keys[i] for i in index]
            return Series({k: self.series[k] for k in keys})
        msg = f"Cannot index with: {index=}"
        raise TypeError(msg)

    def __setitem__(self, index: int | slice | list[int], value: Any) -> Any:
        if isinstance(index, int):
            self.series[self.keys[index]] = value
            return
        if isinstance(index, slice):
            for k in self.keys[index]:
                self.series[k] = value
            return
        if isinstance(index, list):
            for i in index:
                self.series[self.keys[i]] = value
            return
        msg = f"Cannot index with: {index=}"
        raise TypeError(msg)


class Series(UserDict):
    name: Hashable
    iloc_indexer: IlocIndexer

    ###########################################################################
    # Initializer and general methods
    ###########################################################################
    def __init__(
        self, data: Mapping | Collection | Scalar | None = None, keys: Keys | None = None, name: Hashable = None
    ):
        if data is None:
            super().__init__()
        elif isinstance(data, Mapping):
            if keys is not None:
                data = {k: v for k, v in data.items() if k in keys}
            super().__init__(data)
        elif isinstance(data, Scalar):
            super().__init__({0: data})
        elif isinstance(data, Collection):
            if keys is None:
                keys = range(len(data))
            elif len(data) != len(list(keys)):
                msg = "Length of data and keys must match"
                raise ValueError(msg)
            super().__init__(dict(zip(keys, data)))
        else:
            msg = f"Unexpected data type: {type(data)=}"
            raise ValueError(msg)
        self.name = name

    def copy(self, *, deep: bool = True):
        if deep:
            return copy.deepcopy(self)
        return copy.copy(self)

    def rename(self, name: Hashable):
        self.name = name

    ###########################################################################
    # Accessors
    ###########################################################################
    @property
    def loc(self) -> LocIndexer:
        if not hasattr(self, "loc_indexer"):
            self.loc_indexer = LocIndexer(self)
        return self.loc_indexer

    @property
    def iloc(self) -> IlocIndexer:
        if not hasattr(self, "iloc_indexer"):
            self.iloc_indexer = IlocIndexer(self)
        return self.iloc_indexer

    def head(self, n: int = 5) -> Series:
        return self.iloc[:n]

    def tail(self, n: int = 5) -> Series:
        return self.iloc[-n:]

    ###########################################################################
    # Operations and Comparisons Auxiliary Functions
    ###########################################################################
    def _other_as_series(self, other: Series | Scalar | Collection) -> Series:
        if isinstance(other, Series):
            return other
        if isinstance(other, Scalar):
            return Series([other] * len(self), keys=self.keys())
        if isinstance(other, Collection):
            return Series(other, keys=self.keys())
        return NotImplemented

    def _match_keys(self, other: Series):
        if self.keys() != other.keys():
            msg = "Cannot operate in Series with different keys"
            raise KeyError(msg)

    def _other_as_series_matching(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series(other)
        self._match_keys(other)
        return other

    ###########################################################################
    # Map/Reduce
    ###########################################################################
    def map(self, func: Callable) -> Series:
        return Series({k: func(v) for k, v in self.items()})

    def reduce(self, func: Callable, initial: Any):
        if len(self) > 0:
            return reduce(func, self.items(), initial)
        return initial

    def max(self):
        return max(self.values())

    def min(self):
        return min(self.values())

    def sum(self):
        return sum(self.values())

    def all(self):
        return all(self.values())

    def any(self):
        return any(self.values())

    def argmax(self):
        return self.reduce(lambda acc, cur: acc if acc[1] > cur[1] else cur, (next(iter(self.keys())), self.iloc[0]))[0]

    def argmin(self):
        return self.reduce(lambda acc, cur: acc if acc[1] < cur[1] else cur, (next(iter(self.keys())), self.iloc[0]))[0]

    ###########################################################################
    # Statistics
    ###########################################################################
    def mean(self):
        return statistics.mean(self.values())

    def fmean(self, weights=None):
        return statistics.fmean(self.values(), weights)

    def geometric_mean(self):
        return statistics.geometric_mean(self.values())

    def harmonic_mean(self, weights=None):
        return statistics.harmonic_mean(self.values(), weights)

    # def kde(self):
    #     return statistics.kde(self.values())

    # def kde_random(self):
    #     return statistics.kde_random(self.values())

    def median(self):
        return statistics.median(self.values())

    def median_low(self):
        return statistics.median_low(self.values())

    def median_high(self):
        return statistics.median_high(self.values())

    def median_grouped(self):
        return statistics.median_grouped(self.values())

    def mode(self):
        return statistics.mode(self.values())

    def multimode(self):
        return statistics.multimode(self.values())

    def quantiles(self, *, n=4, method: Literal["exclusive", "inclusive"] = "exclusive"):
        return statistics.quantiles(self.values(), n=n, method=method)

    def pstdev(self, mu=None):
        return statistics.pstdev(self.values(), mu)

    def pvariance(self, mu=None):
        return statistics.pvariance(self.values(), mu)

    def stdev(self, xbar=None):
        return statistics.stdev(self.values(), xbar)

    def variance(self, xbar=None):
        return statistics.variance(self.values(), xbar)

    def covariance(self, other: Series, /):
        self._match_keys(other)
        x = list(self.values())
        y = [other[k] for k in self.keys()]
        return statistics.covariance(x, y)

    # def correlation(self, other: Series, /, *, method: Literal["linear"] = "linear"):
    def correlation(self, other: Series, /):
        self._match_keys(other)
        x = list(self.values())
        y = [other[k] for k in self.keys()]
        return statistics.correlation(x, y)

    def linear_regression(self, other: Series, /, *, proportional: bool = False):
        self._match_keys(other)
        x = list(self.values())
        y = [other[k] for k in self.keys()]
        return statistics.linear_regression(x, y, proportional=proportional)

    ###########################################################################
    # Exports
    ###########################################################################
    def to_list(self) -> list[Any]:
        return list(self.values())

    def to_dict(self) -> dict[Hashable, Any]:
        return dict(self)

    ###########################################################################
    # Comparisons
    ###########################################################################
    def __lt__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return Series({k: v < other[k] for k, v in self.items()})

    def __le__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return Series({k: v <= other[k] for k, v in self.items()})

    def __eq__(self, other: Series | Collection | Scalar) -> Series:  # type: ignore
        other = self._other_as_series_matching(other)
        return Series({k: v == other[k] for k, v in self.items()})

    def __ne__(self, other: Series | Collection | Scalar) -> Series:  # type: ignore
        other = self._other_as_series_matching(other)
        return Series({k: v != other[k] for k, v in self.items()})

    def __gt__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return Series({k: v > other[k] for k, v in self.items()})

    def __ge__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return Series({k: v >= other[k] for k, v in self.items()})

    ###########################################################################
    # Operators
    ###########################################################################
    def __add__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return Series({k: v + other[k] for k, v in self.items()})

    def __sub__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return Series({k: v - other[k] for k, v in self.items()})

    def __mul__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return Series({k: v * other[k] for k, v in self.items()})

    def __matmul__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return Series({k: v @ other[k] for k, v in self.items()})

    def __truediv__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return Series({k: v / other[k] for k, v in self.items()})

    def __floordiv__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return Series({k: v // other[k] for k, v in self.items()})

    def __mod__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return Series({k: v % other[k] for k, v in self.items()})

    def __divmod__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return Series({k: divmod(v, other[k]) for k, v in self.items()})

    def __pow__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return Series({k: pow(v, other[k]) for k, v in self.items()})

    def __lshift__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return Series({k: v << other[k] for k, v in self.items()})

    def __rshift__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return Series({k: v >> other[k] for k, v in self.items()})

    def __and__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return Series({k: v & other[k] for k, v in self.items()})

    def __xor__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return Series({k: v ^ other[k] for k, v in self.items()})

    def __or__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return Series({k: v | other[k] for k, v in self.items()})

    def __radd__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return other + self

    def __rsub__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return other - self

    def __rmul__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return other * self

    def __rtruediv__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return other / self

    def __rfloordiv__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return other // self

    def __rmod__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return other % self

    def __rdivmod__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return divmod(other, self)

    def __rpow__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return pow(other, self)

    def __rlshift__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return other << self

    def __rrshift__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return other >> self

    def __rand__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return other & self

    def __rxor__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return other ^ self

    def __ror__(self, other: Series | Collection | Scalar) -> Series:
        other = self._other_as_series_matching(other)
        return other | self

    def __iadd__(self, other: Series | Collection | Scalar):
        other = self._other_as_series_matching(other)
        for k in self:
            self[k] += other[k]

    def __isub__(self, other: Series | Collection | Scalar):
        other = self._other_as_series_matching(other)
        for k in self:
            self[k] -= other[k]

    def __imul__(self, other: Series | Collection | Scalar):
        other = self._other_as_series_matching(other)
        for k in self:
            self[k] *= other[k]

    def __itruediv__(self, other: Series | Collection | Scalar):
        other = self._other_as_series_matching(other)
        for k in self:
            self[k] /= other[k]

    def __ifloordiv__(self, other: Series | Collection | Scalar):
        other = self._other_as_series_matching(other)
        for k in self:
            self[k] //= other[k]

    def __imod__(self, other: Series | Collection | Scalar):
        other = self._other_as_series_matching(other)
        for k in self:
            self[k] %= other[k]

    def __ipow__(self, other: Series | Collection | Scalar):
        other = self._other_as_series_matching(other)
        for k in self:
            self[k] **= other[k]

    def __ilshift__(self, other: Series | Collection | Scalar):
        other = self._other_as_series_matching(other)
        for k in self:
            self[k] <<= other[k]

    def __irshift__(self, other: Series | Collection | Scalar):
        other = self._other_as_series_matching(other)
        for k in self:
            self[k] >>= other[k]

    def __iand__(self, other: Series | Collection | Scalar):
        other = self._other_as_series_matching(other)
        for k in self:
            self[k] &= other[k]

    def __ixor__(self, other: Series | Collection | Scalar):
        other = self._other_as_series_matching(other)
        for k in self:
            self[k] ^= other[k]

    def __ior__(self, other: Series | Collection | Scalar):  # type: ignore
        other = self._other_as_series_matching(other)
        for k in self:
            self[k] |= other[k]

    ###########################################################################
    # Unary Operators
    ###########################################################################
    def __neg__(self):
        return Series({k: -v for k, v in self.items()})

    def __pos__(self):
        return Series({k: +v for k, v in self.items()})

    def __abs__(self):
        return Series({k: abs(v) for k, v in self.items()})

    def __invert__(self):
        return Series({k: ~v for k, v in self.items()})
