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
Index: TypeAlias = Collection[Hashable]


class LocIndexer:
    def __init__(self, series: Series):
        self.series = series

    def __getitem__(self, key: Hashable | list[Hashable] | Series) -> Any:
        if isinstance(key, list):
            return Series({k: self.series[k] for k in key})
        if isinstance(key, Series):
            if self._is_boolean_mask(key):
                return Series({k: v for k, v in self.series.items() if key[k]})
            return Series({k: self.series[k] for k in key.values})
        if isinstance(key, Hashable):
            return self.series[key]
        msg = f"Cannot index with unhashable: {key=}"
        raise TypeError(msg)

    def __setitem__(self, key: Hashable | list[Hashable], value: Any) -> Any:
        if isinstance(key, list):
            for k in key:
                self.series[k] = value
            return
        if isinstance(key, Hashable):
            self.series[key] = value
            return
        msg = f"Cannot index with unhashable: {key=}"
        raise TypeError(msg)

    def _is_boolean_mask(self, s: Series) -> bool:
        return self.series._match_index(s) and s.map(lambda v: isinstance(v, bool)).all()  # noqa: SLF001


class IlocIndexer:
    def __init__(self, series: Series):
        self.series = series
        self.index = list(series.index)

    def __getitem__(self, index: int | slice | list[int]) -> Any:
        if isinstance(index, int):
            return self.series[self.index[index]]
        if isinstance(index, slice):
            label_index = self.index[index]
            return Series({k: self.series[k] for k in label_index})
        if isinstance(index, list):
            label_index = [self.index[i] for i in index]
            return Series({k: self.series[k] for k in label_index})
        msg = f"Cannot index with: {index=}"
        raise TypeError(msg)

    def __setitem__(self, index: int | slice | list[int], value: Any) -> Any:
        if isinstance(index, int):
            self.series[self.index[index]] = value
            return
        if isinstance(index, slice):
            for k in self.index[index]:
                self.series[k] = value
            return
        if isinstance(index, list):
            for i in index:
                self.series[self.index[i]] = value
            return
        msg = f"Cannot index with: {index=}"
        raise TypeError(msg)


class Series(UserDict):
    """
    Series class representing a one-dimensional labeled array with capabilities for data analysis.

    Attributes:
        name (Hashable): Name of the Series.
        loc_indexer (LocIndexer): Indexer for label-based location selection.
        iloc_indexer (ilocIndexer): Indexer for integer-based location selection.
    """

    name: Hashable
    loc_indexer: LocIndexer
    iloc_indexer: IlocIndexer
    __slots__ = ("name", "loc_indexer", "iloc_indexer")

    ###########################################################################
    # Initializer and general methods
    ###########################################################################
    def __init__(
        self, data: Mapping | Collection | Scalar | None = None, index: Index | None = None, name: Hashable = None
    ):
        """
        Initializes a Series object.

        Args:
            data (Mapping | Collection | Scalar, optional): Data for the Series. Can be a dictionary, list, or scalar. Defaults to None.
            index (Index, optional): Index for the Series. Defaults to None.
            name (Hashable, optional): Name to assign to the Series. Defaults to None.

        Raises:
            ValueError: If the length of data and index don't match, or if data type is unexpected.
        """
        if data is None:
            super().__init__()
        elif isinstance(data, Mapping):
            if index is not None:
                data = {k: v for k, v in data.items() if k in index}
            super().__init__(data)
        elif isinstance(data, Scalar):
            super().__init__({0: data})
        elif isinstance(data, Collection):
            if index is None:
                index = range(len(data))
            elif len(data) != len(list(index)):
                msg = f"Length of values ({len(data)}) does not match length of index ({len(index)})"
                raise ValueError(msg)
            super().__init__(dict(zip(index, data)))
        else:
            msg = f"Unexpected data type: {type(data)=}"
            raise ValueError(msg)
        self.name = name
        self._set_indexers()

    def _set_indexers(self):
        self.iloc = IlocIndexer(self)
        self.loc = LocIndexer(self)

    def copy(self, *, deep: bool = True):
        """
        Creates a copy of the Series.

        Args:
            deep (bool, optional): If True, creates a deep copy. Otherwise, creates a shallow copy. Defaults to True.

        Returns:
            Series: A copy of the Series.
        """
        clone = copy.deepcopy(self) if deep else copy.copy(self)
        clone._set_indexers()  # noqa: SLF001
        return clone

    def rename(self, name: Hashable) -> Series:
        """
        Renames the Series.

        Args:
            name (Hashable): The new name for the Series.

        Returns:
            Series: A new Series with the updated name (a copy).
        """
        clone = self.copy(deep=True)
        clone.name = name
        return clone

    @property
    def index(self) -> Index:
        """
        Returns the index of the Series.

        Returns:
            Index: The index of the Series.
        """
        return list(self.keys())

    @index.setter
    def index(self, index: Index):
        """
        Sets the index of the Series.

        Args:
            value (Index): The new index for the Series.

        Raises:
            ValueError: If the length of the new index does not match the length of the Series.
        """
        if len(self) != len(index):
            msg = f"Length mismatch: Expected axis has {len(self)} elements, new values have {len(index)} elements"
            raise ValueError(msg)
        self.data = dict(zip(index, self.values))

    @property
    def values(self) -> list[Any]:  # type: ignore
        """
        Returns the index of the Series.

        Returns:
            Index: The index of the Series.
        """
        return list(self.data.values())

    ###########################################################################
    # Accessors
    ###########################################################################
    def __getitem__(self, name: Hashable | list[Hashable] | slice | Series) -> Any | Series:
        """
        Retrieves an item or slice from the Series.

        Args:
            name (Hashable | list[Hashable] | slice): The key, list of keys, or slice to retrieve.

        Returns:
            Any: The value(s) associated with the given key(s) or slice.
            Series: A new Series if a list or slice is provided.
        """
        if isinstance(name, (list, Series)):
            return self.loc[name]
        if isinstance(name, slice):
            return self.iloc[name]
        return super().__getitem__(name)

    def head(self, n: int = 5) -> Series:
        """
        Returns the first n rows.

        Args:
            n (int, optional): Number of rows to return. Defaults to 5.

        Returns:
            Series: A new Series containing the first n rows.
        """
        return self.iloc[:n]

    def tail(self, n: int = 5) -> Series:
        """
        Returns the last n rows.

        Args:
            n (int, optional): Number of rows to return. Defaults to 5.

        Returns:
            Series: A new Series containing the last n rows.
        """
        return self.iloc[-n:]

    def ifind(self, val: Any) -> int | None:
        """
        Finds the first integer position (index) of a given value in the Series.

        Args:
            val (Any): The value to search for.

        Returns:
            int | None: The integer position (index) of the first occurrence of the value,
                        or None if the value is not found.
        """
        for i, v in enumerate(self.values):
            if v == val:
                return i
        return None

    def find(self, val: Any) -> Hashable | None:
        """
        Finds the first label (key) associated with a given value in the Series.

        Args:
            val (Any): The value to search for.

        Returns:
            Hashable | None: The label (key) of the first occurrence of the value,
                             or None if the value is not found.
        """
        for k, v in self.items():
            if v == val:
                return k
        return None

    ###########################################################################
    # Operations and Comparisons Auxiliary Functions
    ###########################################################################
    def _other_as_series(self, other: Series | Scalar | Collection) -> Series:
        """Converts other to a Series if it is not already. Used for operations."""
        if isinstance(other, Series):
            return other
        if isinstance(other, Scalar):
            return Series([other] * len(self), index=self.index)
        if isinstance(other, Collection):
            return Series(other, index=self.index)
        return NotImplemented  # no cov

    def _match_index(self, other: Series) -> bool:
        """Checks if the index of other matches the index of self. Used for operations."""
        return self.index == other.index

    def _other_as_series_matching(self, other: Series | Collection | Scalar) -> Series:
        """Converts and matches index of other to self. Used for operations."""
        other = self._other_as_series(other)
        if not self._match_index(other):
            msg = "Cannot operate in Series with different index"
            raise ValueError(msg)
        return other

    ###########################################################################
    # Map/Reduce
    ###########################################################################
    def map(self, func: Callable) -> Series:
        """
        Applies a function to each value in the Series.

        Args:
            func (Callable): The function to apply.

        Returns:
            Series: A new Series with the results of the function applied.
        """
        return Series({k: func(v) for k, v in self.items()})

    def reduce(self, func: Callable, initial: Any):
        """
        Reduces the Series using a function.

        Args:
            func (Callable): The function to apply for reduction.
            initial (Any): The initial value for the reduction.

        Returns:
            Any: The reduced value.
        """
        if len(self) > 0:
            return reduce(func, self.items(), initial)
        return initial

    def agg(self, func: Callable) -> Any:
        """
        Applies an aggregation function to the Series' values.

        This method applies a given function to all the values in the Series.
        It is intended for aggregation functions that operate on a collection
        of values and return a single result.

        Args:
            func (Callable): The aggregation function to apply. This function
                should accept an iterable (like a list or NumPy array) and
                return a single value.

        Returns:
            Any: The result of applying the aggregation function to the Series' values.
        """
        return func(self.values)

    def astype(self, new_type: type) -> Series:
        """
        Casts the Series to a new type.

        Args:
            new_type (type): The type to cast to.

        Returns:
            Series: A new Series with the values cast to the new type.
        """
        return self.copy(deep=True).map(new_type)

    def max(self):
        """
        Returns the maximum value in the Series.

        Returns:
            Any: The maximum value.
        """
        return self.agg(max)

    def min(self):
        """
        Returns the minimum value in the Series.

        Returns:
            Any: The minimum value.
        """
        return self.agg(min)

    def sum(self):
        """
        Returns the sum of the values in the Series.

        Returns:
            Any: The sum of the values.
        """
        return self.agg(sum)

    def all(self):
        """
        Returns True if all values in the Series are True.

        Returns:
            bool: True if all values are True, False otherwise.
        """
        return self.agg(all)

    def any(self):
        """
        Returns True if any value in the Series is True.

        Returns:
            bool: True if any value is True, False otherwise.
        """
        return self.agg(any)

    def argmax(self):
        """
        Returns the index of the maximum value.

        Returns:
            int: The index of the maximum value.
        """
        if len(self) == 0:
            msg = "Attempt to get argmax of an empty sequence"
            raise ValueError(msg)
        return self.ifind(self.max())

    def argmin(self):
        """
        Returns the index of the minimum value.

        Returns:
            int: The index of the minimum value.
        """
        if len(self) == 0:
            msg = "Attempt to get argmin of an empty sequence"
            raise ValueError(msg)
        return self.ifind(self.min())

    def idxmax(self):
        """
        Returns the label of the maximum value.

        Returns:
            Hashable: The label of the maximum value.
        """
        if len(self) == 0:
            msg = "Attempt to get ixmax of an empty sequence"
            raise ValueError(msg)
        return self.find(self.max())

    def idxmin(self):
        """
        Returns the label of the minimum value.

        Returns:
            Hashable: The label of the minimum value.
        """
        if len(self) == 0:
            msg = "Attempt to get idxmin of an empty sequence"
            raise ValueError(msg)
        return self.find(self.min())

    ###########################################################################
    # Statistics
    ###########################################################################
    def mean(self):
        """
        Computes the mean of the Series.

        Returns:
            float: Series mean
        """
        return self.agg(statistics.mean)

    def fmean(self, weights=None):
        """
        Convert data to floats and compute the arithmetic mean.

        Returns:
            float: Series mean
        """
        return self.agg(lambda values: statistics.fmean(values, weights))

    def geometric_mean(self):
        """
        Convert data to floats and compute the geometric mean.

        Returns:
            float: Series geometric mean
        """
        return self.agg(statistics.geometric_mean)

    def harmonic_mean(self, weights=None):
        """
        Convert data to floats and compute the harmonic mean.

        Returns:
            float: Series harmonic mean
        """
        return self.agg(lambda values: statistics.harmonic_mean(values, weights))

    def median(self):
        """
        Return the median (middle value) of numeric data, using the common “mean of middle two” method.
        If data is empty, StatisticsError is raised. data can be a sequence or iterable.

        Returns:
            float | int: Series median
        """
        return self.agg(statistics.median)

    def mode(self):
        """
        Return the single most common data point from discrete or nominal data. The mode (when it exists)
        is the most typical value and serves as a measure of central location.

        Returns:
            Any: Series mode
        """
        return self.agg(statistics.mode)

    def multimode(self):
        """
        Return a list of the most frequently occurring values in the order they were first encountered
        in the data. Will return more than one result if there are multiple modes or an empty list if
        the data is empty:

        Returns:
            list[Any]: List containing modes
        """
        return self.agg(statistics.multimode)

    def quantiles(self, *, n=4, method: Literal["exclusive", "inclusive"] = "exclusive"):
        """
        Divide data into n continuous intervals with equal probability. Returns a list of `n - 1`
        cut points separating the intervals.

        Returns:
            list[float]: List containing quantiles
        """
        return self.agg(lambda values: statistics.quantiles(values, n=n, method=method))

    def pstdev(self, mu=None):
        """
        Return the population standard deviation (the square root of the population variance).
        See pvariance() for arguments and other details.

        Returns:
            float: Series population standard deviation
        """
        return self.agg(lambda values: statistics.pstdev(values, mu=mu))

    def pvariance(self, mu=None):
        """
        Return the population variance of data, a non-empty sequence or iterable of real-valued
        numbers. Variance, or second moment about the mean, is a measure of the variability
        (spread or dispersion) of data. A large variance indicates that the data is spread out;
        a small variance indicates it is clustered closely around the mean.

        Returns:
            float: Series population variance
        """
        return self.agg(lambda values: statistics.pvariance(values, mu=mu))

    def stdev(self, xbar=None):
        """
        Return the sample standard deviation (the square root of the sample variance).
        See variance() for arguments and other details.

        Returns:
            float: Series standard deviation
        """
        return self.agg(lambda values: statistics.stdev(values, xbar=xbar))

    def variance(self, xbar=None):
        """
        Return the sample variance of data, an iterable of at least two real-valued numbers.
        Variance, or second moment about the mean, is a measure of the variability
        (spread or dispersion) of data. A large variance indicates that the data is spread out;
        a small variance indicates it is clustered closely around the mean.

        Returns:
            float: Series variance
        """
        return self.agg(lambda values: statistics.variance(values, xbar=xbar))

    # def covariance(self, other: Series, /):
    #     self._match_index(other)
    #     x = list(self.values)
    #     y = [other[k] for k in self.index]
    #     return statistics.covariance(x, y)

    # # def correlation(self, other: Series, /, *, method: Literal["linear"] = "linear"):
    # def correlation(self, other: Series, /):
    #     self._match_index(other)
    #     x = list(self.values)
    #     y = [other[k] for k in self.index]
    #     return statistics.correlation(x, y)

    # def linear_regression(self, other: Series, /, *, proportional: bool = False):
    #     self._match_index(other)
    #     x = list(self.values)
    #     y = [other[k] for k in self.index]
    #     return statistics.linear_regression(x, y, proportional=proportional)

    ###########################################################################
    # Exports
    ###########################################################################
    def to_list(self) -> list[Any]:
        """
        Converts the Series to a list.

        Returns:
            list[Any]: A list of the Series values.
        """
        return self.values

    def to_dict(self) -> dict[Hashable, Any]:
        """
        Converts the Series to a dictionary.

        Returns:
            dict[Hashable, Any]: A dictionary representation of the Series.
        """
        return dict(self)

    ###########################################################################
    # Comparisons
    ###########################################################################
    def __lt__(self, other: Series | Collection | Scalar) -> Series:
        """
        Element-wise less than comparison.

        Compares each value in the Series with the corresponding value in `other`.

        Args:
            other (Series | Collection | Scalar): The other Series, Collection, or Scalar to compare with.

        Returns:
            Series: A Series of boolean values indicating the result of the comparison.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v < other[k] for k, v in self.items()})

    def __le__(self, other: Series | Collection | Scalar) -> Series:
        """
        Element-wise less than or equal to comparison.

        Compares each value in the Series with the corresponding value in `other`.

        Args:
            other (Series | Collection | Scalar): The other Series, Collection, or Scalar to compare with.

        Returns:
            Series: A Series of boolean values indicating the result of the comparison.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v <= other[k] for k, v in self.items()})

    def __eq__(self, other: Series | Collection | Scalar) -> Series:  # type: ignore
        """
        Element-wise equality comparison.

        Compares each value in the Series with the corresponding value in `other`.

        Args:
            other (Series | Collection | Scalar): The other Series, Collection, or Scalar to compare with.

        Returns:
            Series: A Series of boolean values indicating the result of the comparison.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v == other[k] for k, v in self.items()})

    def __ne__(self, other: Series | Collection | Scalar) -> Series:  # type: ignore
        """
        Element-wise inequality comparison.

        Compares each value in the Series with the corresponding value in `other`.

        Args:
            other (Series | Collection | Scalar): The other Series, Collection, or Scalar to compare with.

        Returns:
            Series: A Series of boolean values indicating the result of the comparison.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v != other[k] for k, v in self.items()})

    def __gt__(self, other: Series | Collection | Scalar) -> Series:
        """
        Element-wise greater than comparison.

        Compares each value in the Series with the corresponding value in `other`.

        Args:
            other (Series | Collection | Scalar): The other Series, Collection, or Scalar to compare with.

        Returns:
            Series: A Series of boolean values indicating the result of the comparison.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v > other[k] for k, v in self.items()})

    def __ge__(self, other: Series | Collection | Scalar) -> Series:
        """
        Element-wise greater than or equal to comparison.

        Compares each value in the Series with the corresponding value in `other`.

        Args:
            other (Series | Collection | Scalar): The other Series, Collection, or Scalar to compare with.

        Returns:
            Series: A Series of boolean values indicating the result of the comparison.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v >= other[k] for k, v in self.items()})

    ###########################################################################
    # Operators
    ###########################################################################
    def __add__(self, other: Series | Collection | Scalar) -> Series:
        """
        Element-wise addition.

        Args:
            other (Series | Collection | Scalar): The other Series, Collection, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v + other[k] for k, v in self.items()})

    def __sub__(self, other: Series | Collection | Scalar) -> Series:
        """
        Element-wise subtraction.

        Args:
            other (Series | Collection | Scalar): The other Series, Collection, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v - other[k] for k, v in self.items()})

    def __mul__(self, other: Series | Collection | Scalar) -> Series:
        """
        Element-wise multiplication.

        Args:
            other (Series | Collection | Scalar): The other Series, Collection, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v * other[k] for k, v in self.items()})

    def __matmul__(self, other: Series | Collection | Scalar) -> Scalar:
        """
        Performs dot product with another Series, Collection or Scalar.

        If other is a Series or a Collection, performs the dot product between the two.
        If other is a Scalar, multiplies all elements of the Series by the scalar and returns the sum.

        Args:
            other (Series | Collection | Scalar)

        Returns:
            Scalar: The dot product of the Series.
        """
        other = self._other_as_series_matching(other)
        acc = 0
        for key, value in self.items():
            acc += other[key] * value
        return acc

    def __truediv__(self, other: Series | Collection | Scalar) -> Series:
        """
        Element-wise division.

        Args:
            other (Series | Collection | Scalar): The other Series, Collection, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v / other[k] for k, v in self.items()})

    def __floordiv__(self, other: Series | Collection | Scalar) -> Series:
        """
        Element-wise floor division.

        Args:
            other (Series | Collection | Scalar): The other Series, Collection, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v // other[k] for k, v in self.items()})

    def __mod__(self, other: Series | Collection | Scalar) -> Series:
        """
        Element-wise modulo.

        Args:
            other (Series | Collection | Scalar): The other Series, Collection, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v % other[k] for k, v in self.items()})

    def __divmod__(self, other: Series | Collection | Scalar) -> Series:
        """
        Element-wise divmod.

        Args:
            other (Series | Collection | Scalar): The other Series, Collection, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = self._other_as_series_matching(other)
        return Series({k: divmod(v, other[k]) for k, v in self.items()})

    def __pow__(self, other: Series | Collection | Scalar) -> Series:
        """
        Element-wise exponentiation.

        Args:
            other (Series | Collection | Scalar): The other Series, Collection, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = self._other_as_series_matching(other)
        return Series({k: pow(v, other[k]) for k, v in self.items()})

    def __lshift__(self, other: Series | Collection | Scalar) -> Series:
        """
        Element-wise left bit shift.

        Args:
            other (Series | Collection | Scalar): The other Series, Collection, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v << other[k] for k, v in self.items()})

    def __rshift__(self, other: Series | Collection | Scalar) -> Series:
        """
        Element-wise right bit shift.

        Args:
            other (Series | Collection | Scalar): The other Series, Collection, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v >> other[k] for k, v in self.items()})

    def __and__(self, other: Series | Collection | Scalar) -> Series:
        """
        Element-wise AND.

        Args:
            other (Series | Collection | Scalar): The other Series, Collection, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v & other[k] for k, v in self.items()})

    def __xor__(self, other: Series | Collection | Scalar) -> Series:
        """
        Element-wise XOR.

        Args:
            other (Series | Collection | Scalar): The other Series, Collection, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v ^ other[k] for k, v in self.items()})

    def __or__(self, other: Series | Collection | Scalar) -> Series:
        """
        Element-wise OR.

        Args:
            other (Series | Collection | Scalar): The other Series, Collection, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = self._other_as_series_matching(other)
        return Series({k: v | other[k] for k, v in self.items()})

    ###########################################################################
    # Right-hand Side Operators
    ###########################################################################
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

    ###########################################################################
    # In-place Operators
    ###########################################################################
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

    # def __imatmul__(self, other: Series | Collection | Scalar):
    #     other = self._other_as_series_matching(other)
    #     for k in self:
    #         self[k] @= other[k]

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
