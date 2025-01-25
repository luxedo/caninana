# SPDX-FileCopyrightText: 2025-present Luiz Eduardo Amaral <luizamaral306@gmail.com>
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import copy
import statistics
from collections import UserDict
from collections.abc import Callable, Collection, Generator, Hashable, Iterator, Mapping, Sequence
from functools import reduce
from typing import Any, Literal, TypeAlias, overload

AxisRows = 0
AxisCols = 1
Scalar: TypeAlias = int | float | complex | str | bool
Index: TypeAlias = Sequence[Hashable]
Axis: TypeAlias = Literal[0, 1]
AxisOrNone: TypeAlias = Axis | None


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

    def __repr__(self) -> str:
        if len(self) == 0:
            if self.name is None:
                return "Empty Series"
            return f'Empty Series(name="{self.name}")'
        columns = list(zip(*self.items()))
        widths = [max([len(str(v)) for v in col]) for col in columns]
        height = len(columns[0])
        ret = [[f"{col[i]!s:>{width}}" for col, width in zip(columns, widths)] for i in range(height)]
        return "\n".join("  ".join(r) for r in ret) + f"\nname: {self.name}\n"

    def copy(self, *, deep: bool = True):
        """
        Creates a copy of the Series.

        Args:
            deep (bool, optional): If True, creates a deep copy. Otherwise, creates a shallow copy. Defaults to True.

        Returns:
            Series: A copy of the Series.
        """
        clone = copy.deepcopy(self) if deep else copy.copy(self)
        clone.name = self.name
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

    def reindex(self, index: Index) -> Series:
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
        clone = self.copy(deep=True)
        clone.data = dict(zip(index, self.values))
        return clone

    @property
    def values(self) -> list[Any]:  # type: ignore
        """
        Return a list representation of the Series.

        Returns:
            list: The values of the Series.
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
    # Auxiliary Functions
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
        return self.map(new_type)

    def abs(self) -> Series:
        """
        Returns the absolute values for Series

        Returns:
            Series: Absolute values Series
        """
        return self.map(abs)

    def dot(self, other: Series | Collection | Scalar) -> Scalar:
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

    def max(self) -> Scalar:
        """
        Returns the maximum value in the Series.

        Returns:
            Any: The maximum value.
        """
        return self.agg(max)

    def min(self) -> Scalar:
        """
        Returns the minimum value in the Series.

        Returns:
            Any: The minimum value.
        """
        return self.agg(min)

    def sum(self) -> Scalar:
        """
        Returns the sum of the values in the Series.

        Returns:
            Any: The sum of the values.
        """
        return self.agg(sum)

    def all(self) -> bool:
        """
        Returns True if all values in the Series are True.

        Returns:
            bool: True if all values are True, False otherwise.
        """
        return self.agg(all)

    def any(self) -> bool:
        """
        Returns True if any value in the Series is True.

        Returns:
            bool: True if any value is True, False otherwise.
        """
        return self.agg(any)

    def argmax(self) -> int:
        """
        Returns the index of the maximum value.

        Returns:
            int: The index of the maximum value.
        """
        if len(self) == 0:
            msg = "Attempt to get argmax of an empty sequence"
            raise ValueError(msg)
        return self.ifind(self.max())  # type: ignore

    def argmin(self) -> int:
        """
        Returns the index of the minimum value.

        Returns:
            int: The index of the minimum value.
        """
        if len(self) == 0:
            msg = "Attempt to get argmin of an empty sequence"
            raise ValueError(msg)
        return self.ifind(self.min())  # type: ignore

    def idxmax(self) -> Hashable | None:
        """
        Returns the label of the maximum value.

        Returns:
            Hashable: The label of the maximum value.
        """
        if len(self) == 0:
            msg = "Attempt to get ixmax of an empty sequence"
            raise ValueError(msg)
        return self.find(self.max())

    def idxmin(self) -> Hashable | None:
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
    def mean(self) -> Scalar:
        """
        Computes the mean of the Series.

        Returns:
            float: Series mean
        """
        return self.agg(statistics.mean)

    def median(self) -> Scalar:
        """
        Return the median (middle value) of numeric data, using the common “mean of middle two” method.
        If data is empty, StatisticsError is raised. data can be a sequence or iterable.

        Returns:
            float | int: Series median
        """
        return self.agg(statistics.median)

    def mode(self) -> Scalar:
        """
        Return the single most common data point from discrete or nominal data. The mode (when it exists)
        is the most typical value and serves as a measure of central location.

        Returns:
            Any: Series mode
        """
        return self.agg(statistics.mode)

    def quantiles(self, *, n=4, method: Literal["exclusive", "inclusive"] = "exclusive") -> Collection[float]:
        """
        Divide data into n continuous intervals with equal probability. Returns a list of `n - 1`
        cut points separating the intervals.

        Returns:
            list[float]: List containing quantiles
        """
        return self.agg(lambda values: statistics.quantiles(values, n=n, method=method))

    def std(self, xbar=None) -> Scalar:
        """
        Return the sample standard deviation (the square root of the sample variance).
        See variance() for arguments and other details.

        Returns:
            float: Series standard deviation
        """
        return self.agg(lambda values: statistics.stdev(values, xbar=xbar))

    def var(self, xbar=None) -> Scalar:
        """
        Return the sample variance of data, an iterable of at least two real-valued numbers.
        Variance, or second moment about the mean, is a measure of the variability
        (spread or dispersion) of data. A large variance indicates that the data is spread out;
        a small variance indicates it is clustered closely around the mean.

        Returns:
            float: Series variance
        """
        return self.agg(lambda values: statistics.variance(values, xbar=xbar))

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
        return self.dot(other)

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

    def __imatmul__(self, other: Series | Collection | Scalar):
        other = self._other_as_series_matching(other)
        for k in self:
            self[k] @= other[k]

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
    def __neg__(self) -> Series:
        return Series({k: -v for k, v in self.items()})

    def __pos__(self) -> Series:
        return Series({k: +v for k, v in self.items()})

    def __abs__(self) -> Series:
        return self.abs()

    def __invert__(self) -> Series:
        return Series({k: ~v for k, v in self.items()})


class DataFrame(UserDict):
    index: Index
    columns: Index
    loc_indexer: LocIndexer
    iloc_indexer: IlocIndexer
    _length: int
    __slots__ = ("name", "index", "loc_indexer", "iloc_indexer", "_length")

    ###########################################################################
    # Initializer and general methods
    ###########################################################################
    def __init__(
        self,
        data: Mapping[Hashable, Series]
        | Mapping[Hashable, Collection[Scalar]]
        | Collection[Series]
        | Collection[Mapping[Hashable, Scalar]]
        | Collection[Scalar]
        | Collection[Collection[Scalar]]
        | Iterator
        | None = None,
        index: Index | None = None,
        columns: Index | None = None,
    ):
        """
        Initializes a DataFrame object.

        Args:
            data (Mapping | Collection | Scalar, optional): Data for the Series. Can be a dictionary, list, or scalar. Defaults to None.
            index (Index, optional): Index for the Series. Defaults to None.
            name (Hashable, optional): Name to assign to the Series. Defaults to None.

        Raises:
            ValueError: If the length of data and index don't match, or if data type is unexpected.
        """
        if isinstance(data, Iterator):
            data = list(data)
        match data:
            case None:
                self._init_empty(index, columns)
            case Mapping() | Collection() if len(data) == 0:
                self._init_empty(index, columns)
            case Mapping() as m if all(isinstance(v, (Series, Collection)) for v in m.values()):
                self._init_mapping_of_series({col: Series(val) for col, val in data.items()}, index, columns)  # type: ignore
            case Collection() as c if all(isinstance(v, Series) for v in c):
                self._init_collection_of_series(data, index, columns)  # type: ignore
            case Collection() as c if all(isinstance(v, Mapping) for v in c):
                self._init_collection_of_series([Series(row) for row in data], index, columns)  # type: ignore
            case Collection() as c if all(isinstance(v, Scalar) for v in c) and not isinstance(c, str):
                self._init_mapping_of_series({0: Series(data)}, index, columns)
            case Collection() as c if all(isinstance(v, Collection) for v in c) and not isinstance(c, str):
                self._init_collection_of_series([Series(row) for row in data], index, columns)  # type: ignore
            case _:
                msg = "DataFrame constructor not properly called!"
                raise ValueError(msg)
        self._validate_index_and_columns()

    def _init_empty(self, index: Index | None = None, columns: Index | None = None):
        super().__init__()
        if (index is not None and len(index) > 0) or (columns is not None and len(columns) > 0):
            msg = "Cannot create an empty DataFrame with preset columns and/or indexes"
            raise ValueError(msg)
        self.index = []
        self.columns = []

    def _init_mapping_of_series(
        self, data: Mapping[Hashable, Series], index: Index | None = None, columns: Index | None = None
    ):
        col0 = next(iter(data))
        val0 = data[col0]
        if len(val0) == 0:
            self._init_empty(index, columns)
            return

        self.index = val0.index if index is None else index
        self.columns = list(data.keys()) if columns is None else columns
        if (len(self.index) != len(val0.index)) or (len(self.columns) != len(data)):
            passed = (len(val0.index), len(data))
            implied = (len(self.index), len(self.columns))
            msg = f"Shape of passed values is {passed}, indices imply {implied}"
            raise ValueError(msg)
        super().__init__({col: s.copy().rename(col).reindex(self.index) for col, s in zip(self.columns, data.values())})

    def _init_collection_of_series(
        self, data: Collection[Series], index: Index | None = None, columns: Index | None = None
    ):
        row0 = next(iter(data))
        src_columns = row0.index
        if len(src_columns) == 0:
            self._init_empty(index, columns)
            return

        if columns is not None:
            self.columns = columns
        else:
            self.columns = src_columns
        if any(d.index != src_columns for d in data):
            all_cols = {item for d in data for item in d.index}
            missing_cols = all_cols - set(self.columns) or "{}"
            extra_cols = set(self.columns) - all_cols or "{}"
            msg = f"Misaligned columns. Expected {self.columns}. Missing: {missing_cols}, Extra: {extra_cols}"
            raise ValueError(msg)

        # @TODO: Deal with Series with names
        self.index = list(range(len(data))) if index is None else index
        if (len(self.index) != len(data)) or (len(self.columns) != len(src_columns)):
            passed = (len(data), len(src_columns))
            implied = (len(self.index), len(self.columns))
            msg = f"Shape of passed values is {passed}, indices imply {implied}"
            raise ValueError(msg)
        super().__init__(
            {
                dst_col: Series({idx: row[src_col] for idx, row in zip(self.index, data)}, name=dst_col)
                for src_col, dst_col in zip(src_columns, self.columns)
            }
        )

    def _validate_index_and_columns(self):
        for col, s in self.items():
            if s.index != self.index:
                msg = "Somehow the inner indexes and DataFrame indexex don't match. This shouldn't happen!"
                raise ValueError(msg)
            if s.name != col:
                msg = "Somehow the inner columns and DataFrame columns don't match. This shouldn't happen!"
                raise ValueError(msg)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.index), len(self.columns))

    def __len__(self) -> int:
        return len(self.index)

    def __repr__(self):
        if len(self) == 0:
            return "Empty DataFrame"
        columns = list(zip(["", *self.columns], *[[col, *s.values] for col, s in self.iterrows()]))
        widths = [max([len(str(v)) for v in col]) for col in columns]
        height = len(columns[0])
        ret = [[f"{col[i]!s:>{width}}" for col, width in zip(columns, widths)] for i in range(height)]
        return "\n".join("  ".join(r) for r in ret)

    @property
    def T(self) -> DataFrame:  # noqa: N802
        data = list(self.data.values())
        return DataFrame(data, index=self.columns[:], columns=self.index[:])

    @property
    def values(self) -> list[list[Any]]:  # type: ignore
        """
        Return a list representation of the DataFrame.

        Returns:
            list: The values of the DataFrame.
        """
        return self.to_list()

    def iterrows(self) -> Generator[tuple[Hashable, Series]]:
        yield from self.T.items()

    ###########################################################################
    # Accessors
    ###########################################################################

    ###########################################################################
    # Apply/Agg/Map/Reduce
    ###########################################################################
    def apply(self, method: Callable[[Series], Any], axis: Axis = 0) -> Series:
        match axis:
            case int(c) if c == AxisRows:
                return Series({col: method(s) for col, s in self.items()})
            case int(c) if c == AxisCols:
                return self.T.apply(method, axis=0)
            case _:
                msg = f"No axis named {axis} for object type DataFrame"
                raise ValueError(msg)

    def _apply_with_none(self, method: Callable[[Series], Any], axis: AxisOrNone = 0):
        match axis:
            case None:
                return method(self.apply(method, 0))
            case _:
                return self.apply(method, axis)

    def agg(self, method: Callable[[Collection[Any]], Any], axis: Axis = 0) -> Series:
        match axis:
            case int(c) if c == AxisRows:
                return Series({col: method(s.values) for col, s in self.items()})
            case int(c) if c == AxisCols:
                return self.T.agg(method, axis=0)
            case _:
                msg = f"No axis named {axis} for object type DataFrame"
                raise ValueError(msg)

    def _agg_with_none(self, method: Callable[[Collection[Any]], Any], axis: AxisOrNone = 0):
        match axis:
            case None:
                return method([item for sublist in self.to_list() for item in sublist])
            case _:
                return self.agg(method, axis)

    def map(self, func: Callable) -> DataFrame:
        """
        Applies a function to each value in the DataFrame.

        Args:
            func (Callable): The function to apply.

        Returns:
            DataFrame: A new DataFrame with the results of the function applied.
        """
        return DataFrame({col: s.map(func) for col, s in self.items()})

    def astype(self, new_type: type) -> DataFrame:
        """
        Casts the DataFrame to a new type.

        Args:
            new_type (type): The type to cast to.

        Returns:
            DataFrame: A new DataFrame with the values cast to the new type.
        """
        return self.map(new_type)

    # def dot(self, other: Series | Collection | Scalar) -> DataFrame | Series:
    #     """
    #     @REDO
    #     """
    #     return 10

    def abs(self) -> DataFrame:
        """
        Returns the absolute values for DataFrame

        Returns:
            DataFrame: Absolute values DataFrame
        """
        return self.map(abs)

    @overload
    def max(self) -> Series: ...  # no cov
    @overload
    def max(self, axis: Axis) -> Series: ...  # no cov
    @overload
    def max(self, axis: None) -> Scalar: ...  # no cov
    def max(self, axis: AxisOrNone = 0) -> Series | Scalar:
        return self._apply_with_none(lambda s: s.max(), axis)

    @overload
    def min(self) -> Series: ...  # no cov
    @overload
    def min(self, axis: Axis) -> Series: ...  # no cov
    @overload
    def min(self, axis: None) -> Scalar: ...  # no cov
    def min(self, axis: AxisOrNone = 0) -> Series | Scalar:
        return self._apply_with_none(lambda s: s.min(), axis)

    @overload
    def sum(self) -> Series: ...  # no cov
    @overload
    def sum(self, axis: Axis) -> Series: ...  # no cov
    @overload
    def sum(self, axis: None) -> Scalar: ...  # no cov
    def sum(self, axis: AxisOrNone = 0) -> Series | Scalar:
        return self._apply_with_none(lambda s: s.sum(), axis)

    @overload
    def all(self) -> Series: ...  # no cov
    @overload
    def all(self, axis: Axis) -> Series: ...  # no cov
    @overload
    def all(self, axis: None) -> bool: ...  # no cov
    def all(self, axis: AxisOrNone = 0) -> Series | bool:
        return self._apply_with_none(lambda s: s.all(), axis)

    @overload
    def any(self) -> Series: ...  # no cov
    @overload
    def any(self, axis: Axis) -> Series: ...  # no cov
    @overload
    def any(self, axis: None) -> bool: ...  # no cov
    def any(self, axis: AxisOrNone = 0) -> Series | bool:
        return self._apply_with_none(lambda s: s.any(), axis)

    @overload
    def idxmax(self) -> Series: ...  # no cov
    @overload
    def idxmax(self, axis: Axis) -> Series: ...  # no cov
    @overload
    def idxmax(self, axis: None) -> bool: ...  # no cov
    def idxmax(self, axis: AxisOrNone = 0) -> Series | bool:
        return self._apply_with_none(lambda s: s.idxmax(), axis)

    @overload
    def idxmin(self) -> Series: ...  # no cov
    @overload
    def idxmin(self, axis: Axis) -> Series: ...  # no cov
    @overload
    def idxmin(self, axis: None) -> bool: ...  # no cov
    def idxmin(self, axis: AxisOrNone = 0) -> Series | bool:
        return self._apply_with_none(lambda s: s.idxmin(), axis)

    ###########################################################################
    # Statistics
    ###########################################################################
    @overload
    def mean(self, axis: Axis) -> Series: ...  # no cov
    @overload
    def mean(self, axis: None) -> Scalar: ...  # no cov
    def mean(self, axis: AxisOrNone = 0) -> Series | Scalar:
        """
        Computes the mean of the Series.

        Returns:
            float: Series mean
        """
        return self._agg_with_none(statistics.mean, axis=axis)

    @overload
    def median(self, axis: Axis) -> Series: ...  # no cov
    @overload
    def median(self, axis: None) -> Scalar: ...  # no cov
    def median(self, axis: AxisOrNone = 0) -> Series | Scalar:
        """
        Return the median (middle value) of numeric data, using the common “mean of middle two” method.
        If data is empty, StatisticsError is raised. data can be a sequence or iterable.

        Returns:
            float | int: Series median
        """
        return self._agg_with_none(statistics.median, axis=axis)

    def mode(self, axis: Axis = 0) -> Series:
        """
        Return the single most common data point from discrete or nominal data. The mode (when it exists)
        is the most typical value and serves as a measure of central location.

        Returns:
            Any: Series mode
        """
        # @TOOO: Improve this. Might have to implement NaNs
        return self.agg(statistics.mode, axis=axis)

    def quantiles(self, *, n=4, method: Literal["exclusive", "inclusive"] = "exclusive", axis: Axis = 0) -> Series:
        """
        Divide data into n continuous intervals with equal probability. Returns a list of `n - 1`
        cut points separating the intervals.

        Returns:
            list[float]: List containing quantiles
        """
        return self.agg(lambda values: statistics.quantiles(values, n=n, method=method), axis=axis)

    @overload
    def std(self, xbar, axis: Axis) -> Series: ...  # no cov
    @overload
    def std(self, xbar, axis: None) -> Scalar: ...  # no cov
    def std(self, xbar=None, axis: AxisOrNone = 0) -> Series | Scalar:
        """
        Return the sample standard deviation (the square root of the sample variance).
        See variance() for arguments and other details.

        Returns:
            float: Series standard deviation
        """
        return self._agg_with_none(lambda values: statistics.stdev(values, xbar=xbar), axis=axis)

    @overload
    def var(self, xbar, axis: Axis) -> Series: ...  # no cov
    @overload
    def var(self, xbar, axis: None) -> Scalar: ...  # no cov
    def var(self, xbar=None, axis: AxisOrNone = 0) -> Series | Scalar:
        """
        Return the sample variance of data, an iterable of at least two real-valued numbers.
        Variance, or second moment about the mean, is a measure of the variability
        (spread or dispersion) of data. A large variance indicates that the data is spread out;
        a small variance indicates it is clustered closely around the mean.

        Returns:
            float: Series variance
        """
        return self._agg_with_none(lambda values: statistics.variance(values, xbar=xbar), axis=axis)

    ###########################################################################
    # Exports
    ###########################################################################
    def to_list(self) -> list[list[Any]]:
        """
        Converts the DataFrame to a list.

        Returns:
            list[list[Any]]: A list of the Series values.
        """
        return list(self.apply(lambda s: s.values, axis=1).data.values())

    @overload
    def to_dict(self) -> dict[Hashable, dict[Hashable, Any]]: ...  # no cov
    @overload
    def to_dict(self, orient: Literal["dict"]) -> dict[Hashable, dict[Hashable, Any]]: ...  # no cov
    @overload
    def to_dict(self, orient: Literal["list"]) -> dict[Hashable, list[Any]]: ...  # no cov
    @overload
    def to_dict(self, orient: Literal["records"]) -> list[dict[Hashable, Any]]: ...  # no cov
    def to_dict(self, orient: Literal["dict", "list", "records"] = "dict"):
        """
        Converts the DataFrame to a dictionary.

        Args:
            orient str {`dict`, `list`, `records`}: Determines the type of the values of the
                dictionary.

        Returns:
            dict[Hashable, Any]: A dictionary representation of the Series.
        """
        match orient:
            case "dict":
                return self.apply(lambda s: s.to_dict()).data
            case "list":
                return self.apply(lambda s: s.to_list()).data
            case "records":
                return list(self.apply(lambda s: s.to_dict(), axis=1).values)
            case _:
                msg = f"orient '{orient}' not understood"
                raise ValueError(msg)

    ###########################################################################
    # Comparisons
    ###########################################################################
    def op(
        self, op: str, other: DataFrame | Series | Mapping | Collection | Scalar
    ) -> DataFrame:  # @TODO: Implement axis
        match other:
            case DataFrame():
                return self._op_dataframe(op, other)
            case Series():
                return self._op_series(op, other)
            case Collection() | Mapping() as c if len(c) == 0 and len(self) == 0:
                return DataFrame()
            case Collection() | Mapping() as c if len(c) != len(self.columns):
                msg = f"Unable to coerce to Series, length must be {len(self.columns)}: given {len(other)}"
                raise ValueError(msg)
            case Mapping():
                return self._op_series(op, Series(other))
            case Collection() as c if not isinstance(c, str):
                return self._op_series(op, Series(other, index=self.columns))
            # No 2d collection comparison. Will consider 2d inputs as a series of collections
            case _:  # Everithing else is a scalar then
                return self._op_scalar(op, other)

    def _op_series(self, op: str, other: Series) -> DataFrame:
        if len(self.columns) != len(other):
            msg = "Operands are not aligned. Do `left, right = left.align(right, axis=1, copy=False)` before operating."
            raise ValueError(msg)
        return DataFrame([getattr(row, op)(other) for _, row in self.iterrows()], index=self.index)

    def _op_dataframe(self, op: str, other: DataFrame) -> DataFrame:
        if set(self.keys()) != set(other.keys()):
            msg = "Can only compare identically-labeled (both index and columns) DataFrame objects"
            raise ValueError(msg)
        return DataFrame({col: getattr(s, op)(other[col]) for col, s in self.items()})

    def _op_scalar(self, op: str, other: Collection[Any] | Scalar) -> DataFrame:
        return DataFrame({col: getattr(s, op)(other) for col, s in self.items()})

    def __lt__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:  # type: ignore
        """
        Element-wise less than comparison.

        Compares each value in the DataFrame with the corresponding value in `other`.

        Args:
            other (DataFrame | Series | Collection | Scalar): The other DataFrame Series,
                Collection, or Scalar to compare with.

        Returns:
            DataFrame: A DataFrame of boolean values indicating the result of the comparison.
        """
        return self.op("__lt__", other)

    def __le__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:  # type: ignore
        """
        Element-wise less than or equal to comparison.

        Compares each value in the DataFrame with the corresponding value in `other`.

        Args:
            other (DataFrame | Series | Collection | Scalar): The other DataFrame Series,
                Collection, or Scalar to compare with.

        Returns:
            DataFrame: A DataFrame of boolean values indicating the result of the comparison.
        """
        return self.op("__le__", other)

    def __eq__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:  # type: ignore
        """
        Element-wise equality comparison.

        Compares each value in the DataFrame with the corresponding value in `other`.

        Args:
            other (DataFrame | Series | Collection | Scalar): The other DataFrame Series,
                Collection, or Scalar to compare with.

        Returns:
            DataFrame: A DataFrame of boolean values indicating the result of the comparison.
        """
        return self.op("__eq__", other)

    def __ne__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:  # type: ignore
        """
        Element-wise inequality comparison.

        Compares each value in the DataFrame with the corresponding value in `other`.

        Args:
            other (DataFrame | Series | Collection | Scalar): The other DataFrame Series,
                Collection, or Scalar to compare with.

        Returns:
            DataFrame: A DataFrame of boolean values indicating the result of the comparison.
        """
        return self.op("__ne__", other)

    def __gt__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:  # type: ignore
        """
        Element-wise greater than comparison.

        Compares each value in the DataFrame with the corresponding value in `other`.

        Args:
            other (DataFrame | Series | Collection | Scalar): The other DataFrame Series,
                Collection, or Scalar to compare with.

        Returns:
            DataFrame: A DataFrame of boolean values indicating the result of the comparison.
        """
        return self.op("__gt__", other)

    def __ge__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:  # type: ignore
        """
        Element-wise greater than or equal to comparison.

        Compares each value in the DataFrame with the corresponding value in `other`.

        Args:
            other (DataFrame | Series | Collection | Scalar): The other DataFrame Series,
                Collection, or Scalar to compare with.

        Returns:
            DataFrame: A DataFrame of boolean values indicating the result of the comparison.
        """
        return self.op("__ge__", other)

    ###########################################################################
    # Operators
    ###########################################################################
    def __add__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:
        """
        Element-wise addition.

        Args:
            other (DataFrame | Series | Collection | Scalar): The other DataFrame, Series, Collection, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self.op("__add__", other)

    def __sub__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:
        """
        Element-wise subtraction.

        Args:
            other (DataFrame | Series | Collection | Scalar): The other DataFrame, Series, Collection, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self.op("__sub__", other)

    def __mul__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:
        """
        Element-wise multiplication.

        Args:
            other (DataFrame | Series | Collection | Scalar): The other DataFrame, Series, Collection, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self.op("__mul__", other)

    # def __matmul__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame | Series:
    #     """
    #     @REDO!
    #     """
    #     return self.dot(other)

    def __truediv__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:
        """
        Element-wise division.

        Args:
            other (DataFrame | Series | Collection | Scalar): The other DataFrame, Series, Collection, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self.op("__truediv__", other)

    def __floordiv__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:
        """
        Element-wise floor division.

        Args:
            other (DataFrame | Series | Collection | Scalar): The other DataFrame, Series, Collection, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self.op("__floordiv__", other)

    def __mod__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:
        """
        Element-wise modulo.

        Args:
            other (DataFrame | Series | Collection | Scalar): The other DataFrame, Series, Collection, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self.op("__mod__", other)

    def __divmod__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:
        """
        Element-wise divmod.

        Args:
            other (DataFrame | Series | Collection | Scalar): The other DataFrame, Series, Collection, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self.op("__divmod__", other)

    def __pow__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:
        """
        Element-wise exponentiation.

        Args:
            other (DataFrame | Series | Collection | Scalar): The other DataFrame, Series, Collection, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self.op("__pow__", other)

    def __lshift__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:
        """
        Element-wise left bit shift.

        Args:
            other (DataFrame | Series | Collection | Scalar): The other DataFrame, Series, Collection, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self.op("__lshift__", other)

    def __rshift__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:
        """
        Element-wise right bit shift.

        Args:
            other (DataFrame | Series | Collection | Scalar): The other DataFrame, Series, Collection, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self.op("__rshift__", other)

    def __and__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:
        """
        Element-wise AND.

        Args:
            other (DataFrame | Series | Collection | Scalar): The other DataFrame, Series, Collection, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self.op("__and__", other)

    def __xor__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:
        """
        Element-wise XOR.

        Args:
            other (DataFrame | Series | Collection | Scalar): The other DataFrame, Series, Collection, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self.op("__xor__", other)

    def __or__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:
        """
        Element-wise OR.

        Args:
            other (DataFrame | Series | Collection | Scalar): The other DataFrame, Series, Collection, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self.op("__or__", other)

    ###########################################################################
    # Right-hand Side Operators
    ###########################################################################
    def __radd__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:
        return self.op("__radd__", other)

    def __rsub__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:
        return self.op("__rsub__", other)

    def __rmul__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:
        return self.op("__rmul__", other)

    def __rtruediv__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:
        return self.op("__rtruediv__", other)

    def __rfloordiv__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:
        return self.op("__rfloordiv__", other)

    def __rmod__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:
        return self.op("__rmod__", other)

    def __rdivmod__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:
        return self.op("__rdivmod__", other)

    def __rpow__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:
        return self.op("__rpow__", other)

    def __rlshift__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:
        return self.op("__rlshift__", other)

    def __rrshift__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:
        return self.op("__rrshift__", other)

    def __rand__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:
        return self.op("__rand__", other)

    def __rxor__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:
        return self.op("__rxor__", other)

    def __ror__(self, other: DataFrame | Series | Collection | Scalar) -> DataFrame:
        return self.op("__ror__", other)

    ###########################################################################
    # In-place Operators
    ###########################################################################

    ###########################################################################
    # Unary Operators
    ###########################################################################
    def __neg__(self) -> DataFrame:
        return DataFrame({col: -s for col, s in self.items()})

    def __pos__(self) -> DataFrame:
        return DataFrame({col: +s for col, s in self.items()})

    def __abs__(self) -> DataFrame:
        return self.abs()

    def __invert__(self) -> DataFrame:
        return DataFrame({col: ~s for col, s in self.items()})
