# SPDX-FileCopyrightText: 2025-present Luiz Eduardo Amaral <luizamaral306@gmail.com>
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import copy
import functools
import statistics
from collections import UserList, defaultdict
from collections.abc import Callable, Collection, Generator, Iterator, Mapping, Sequence, Sized
from functools import reduce
from typing import Any, Generic, Literal, Self, TypeAlias, TypeGuard, TypeVar, Union, assert_never, cast, overload

###########################################################################
# Typing
###########################################################################
T = TypeVar("T", "Series", "DataFrame")
AxisRows = 0
AxisCols = 1
Axis: TypeAlias = Literal[0, 1]
AxisOrNone: TypeAlias = Axis | None
Scalar: TypeAlias = int | float | complex | str | bytes | bool
IndexLike: TypeAlias = Sequence[Scalar]
ArrayLike: TypeAlias = Collection
BooleanMask: TypeAlias = Sequence[bool]
LocIndexes: TypeAlias = Union[Scalar, list[Scalar], slice, "Array", "Series"]
IlocIndexes: TypeAlias = Union[int, list[int], slice, "Array", "Series"]
LocDataFrameReturn: TypeAlias = Union["Array", "Series", "DataFrame", Scalar]
LocSeriesReturn: TypeAlias = Union["Series", Scalar]
DfMergeHow: TypeAlias = Literal["inner", "left", "right", "outer"]


def _is_array_like(value: Any) -> TypeGuard[ArrayLike]:
    return isinstance(value, Collection) and not isinstance(value, DataFrame) and not _is_scalar(value)


def _is_scalar(value: Any) -> TypeGuard[Scalar]:
    return isinstance(value, (int, float, complex, str, bool))


def _is_boolean_mask(s: Any) -> TypeGuard[BooleanMask]:
    return isinstance(s, Sequence) and all(isinstance(b, bool) for b in s)


###########################################################################
# Array
###########################################################################
class Array(UserList):
    ###########################################################################
    # Initializer and general methods
    ###########################################################################
    @classmethod
    def full(cls, size: int, fill_value: Scalar) -> Array:
        return cls([fill_value] * size)

    @classmethod
    def zeros(cls, size: int) -> Array:
        return cls.full(size, 0)

    @classmethod
    def ones(cls, size: int) -> Array:
        return cls.full(size, 1)

    ###########################################################################
    # General methods
    ###########################################################################
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.data!s})"

    def copy(self, *, deep: bool = True):
        """
        Creates a copy of the Array.

        Args:
            deep (bool, optional): If True, creates a deep copy. Otherwise, creates a shallow copy. Defaults to True.

        Returns:
            Array: A copy of the Array.
        """
        return copy.deepcopy(self) if deep else copy.copy(self)

    ###########################################################################
    # Accessors
    ###########################################################################
    @overload  # type: ignore
    def __getitem__(self, key: int) -> Scalar: ...  # no cov
    @overload
    def __getitem__(self, key: slice | list[int] | Array) -> Array: ...  # no cov
    def __getitem__(self, key: int | slice | list[int] | Array) -> Scalar | Array:  # type: ignore
        match key:
            case int():
                return self.data[key]
            case slice():
                return Array(self.data[key])
            case Array() | list():
                if _is_boolean_mask(key):
                    return Array([self.data[i] for i, v in enumerate(key) if v])
                return Array([self.data[i] for i in key])
            case _:
                msg = f"Cannot index with: {key=}"
                raise KeyError(msg)

    def __setitem__(self, key: Scalar | slice | ArrayLike, value: Scalar | ArrayLike):  # type: ignore
        indices: list[int]
        match key:
            case int():
                indices = [key]
            case slice():
                indices = list(range(*key.indices(len(self.data))))
            case Array() | list():
                indices = list(key) if not _is_boolean_mask(key) else [i for i, v in enumerate(key) if v]
            case _:
                msg = f"Cannot index with: {key=}"
                raise KeyError(msg)

        match value:
            case v if _is_scalar(v):
                for i in indices:
                    self.data[i] = value
            case values if _is_array_like(values):
                if len(values) != len(indices):
                    msg = "Length of assigned iterable must match the indexes length"
                    raise ValueError(msg)
                for i, v in zip(indices, values):
                    self.data[i] = v
            case _:
                msg = f"Cannot set with: {value=}"
                raise TypeError(msg)

    ###########################################################################
    # Concatenation
    ###########################################################################
    def append(self, value: Scalar) -> Array:  # type: ignore
        """
        Appends `value` to the end of the Array

        Args:
            value (Scalar): The data to append

        Returns:
            Array: A new Array with new data
        """
        super().append(value)
        return self

    ###########################################################################
    # Map/Reduce
    ###########################################################################
    def map(self, func: Callable[[Scalar], Scalar]) -> Array:
        """
        Applies a function to each value in the Array.

        Args:
            func (Callable[Scalar, Any]): The function to apply.

        Returns:
            Array: A new Array with the results of the function applied.
        """
        return Array([func(v) for v in self])

    def reduce(self, func: Callable[[Any, Scalar], Any], initial: Any) -> Any:
        """
        Reduces the Array using a function.

        Args:
            func (Callable[[Any, Scalar], Any]): The function to apply for reduction.
            initial (Any): The initial value for the reduction.

        Returns:
            Any: The reduced value.
        """
        if len(self) > 0:
            return reduce(func, self, initial)
        return initial

    def abs(self) -> Array:
        """
        Returns the absolute values for Array

        Returns:
            Array: Absolute values Array
        """
        return self.map(abs)  # type: ignore

    def dot(self, other: Array | ArrayLike) -> Scalar:
        """
        Performs dot product with another Array, ArrayLike or Scalar.

        If other is a Array or a ArrayLike, performs the dot product between the two.
        If other is a Scalar, multiplies all elements of the Array by the scalar and returns the sum.

        Args:
            other (Array | ArrayLike | Scalar)

        Returns:
            Scalar: The dot product of the Array.
        """
        self._validate_length(other)
        return sum(s * o for s, o in zip(self, other))

    def max(self) -> Scalar:
        """
        Returns the maximum value in the Array.

        Returns:
            Any: The maximum value.
        """
        return max(self)

    def min(self) -> Scalar:
        """
        Returns the minimum value in the Array.

        Returns:
            Any: The minimum value.
        """
        return min(self)

    def sum(self) -> Scalar:
        """
        Returns the sum of the values in the Array.

        Returns:
            Any: The sum of the values.
        """
        return sum(self)

    def all(self) -> bool:
        """
        Returns True if all values in the Array are truthy.

        Returns:
            bool: True if all values are truthy, False otherwise.
        """
        return all(self)

    def any(self) -> bool:
        """
        Returns True if any value in the Array is True.

        Returns:
            bool: True if any value is True, False otherwise.
        """
        return any(self)

    def argmax(self) -> int:
        """
        Returns the index of the maximum value.

        Returns:
            int: The index of the maximum value.
        """
        if len(self) == 0:
            msg = f"Cannot get argmax of an empty {self.__class__.__name__}"
            raise ValueError(msg)
        m = float("-inf")
        a = 0
        for i, v in enumerate(self):
            if v > m:
                m, a = v, i
        return a

    def argmin(self) -> int:
        """
        Returns the index of the minimum value.

        Returns:
            int: The index of the minimum value.
        """
        if len(self) == 0:
            msg = f"Cannot get argmin of an empty {self.__class__.__name__}"
            raise ValueError(msg)
        m = float("inf")
        a = 0
        for i, v in enumerate(self):
            if v < m:
                m, a = v, i
        return a

    ###########################################################################
    # Exports
    ###########################################################################
    def to_list(self) -> list[Any]:
        """
        Converts the Array to a list.

        Returns:
            list[Any]: A list of the Array values.
        """
        return list(self)

    ###########################################################################
    # Comparisons
    ###########################################################################
    @staticmethod
    def _standardize_input(method):
        @functools.wraps(method)
        def wrapper(self: Array, other: ArrayLike | Scalar) -> Array:
            match other:
                case Array():
                    self._validate_length(other)
                case o if _is_array_like(o):
                    other = Array(o)
                    self._validate_length(other)
                case _ as o:
                    other = Array.full(len(self), o)  # type: ignore  # We let any type in
            return method(self, other)  # Call the actual method with standardized input

        return wrapper

    def _validate_length(self, other: Array | ArrayLike):
        if len(self) != len(other):
            msg = f"Cannot operate Arrays with different sizes: {len(self)=}, {len(other)=}"
            raise ValueError(msg)

    @_standardize_input
    def __lt__(self, other: Array | ArrayLike | Scalar) -> Array:
        """
        Element-wise less than comparison.

        Compares each value in the Array with the corresponding value in `other`.

        Args:
            other (Array | ArrayLike | Scalar): The other Array, ArrayLike, or Scalar to compare with.

        Returns:
            Array: A Array of boolean values indicating the result of the comparison.
        """
        other = cast(Array, other)
        return Array([s < o for s, o in zip(self, other)])

    @_standardize_input
    def __le__(self, other: Array | ArrayLike | Scalar) -> Array:
        """
        Element-wise less than or equal to comparison.

        Compares each value in the Array with the corresponding value in `other`.

        Args:
            other (Array | ArrayLike | Scalar): The other Array, ArrayLike, or Scalar to compare with.

        Returns:
            Array: A Array of boolean values indicating the result of the comparison.
        """
        other = cast(Array, other)
        return Array([s <= o for s, o in zip(self, other)])

    @_standardize_input
    def __eq__(self, other: Array | ArrayLike | Scalar) -> Array:  # type: ignore
        """
        Element-wise equality comparison.

        Compares each value in the Array with the corresponding value in `other`.

        Args:
            other (Array | ArrayLike | Scalar): The other Array, ArrayLike, or Scalar to compare with.

        Returns:
            Array: A Array of boolean values indicating the result of the comparison.
        """
        other = cast(Array, other)
        return Array([s == o for s, o in zip(self, other)])

    @_standardize_input
    def __ne__(self, other: Array | ArrayLike | Scalar) -> Array:  # type: ignore
        """
        Element-wise inequality comparison.

        Compares each value in the Array with the corresponding value in `other`.

        Args:
            other (Array | ArrayLike | Scalar): The other Array, ArrayLike, or Scalar to compare with.

        Returns:
            Array: A Array of boolean values indicating the result of the comparison.
        """
        other = cast(Array, other)
        return Array([s != o for s, o in zip(self, other)])

    @_standardize_input
    def __gt__(self, other: Array | ArrayLike | Scalar) -> Array:
        """
        Element-wise greater than comparison.

        Compares each value in the Array with the corresponding value in `other`.

        Args:
            other (Array | ArrayLike | Scalar): The other Array, ArrayLike, or Scalar to compare with.

        Returns:
            Array: A Array of boolean values indicating the result of the comparison.
        """
        other = cast(Array, other)
        return Array([s > o for s, o in zip(self, other)])

    @_standardize_input
    def __ge__(self, other: Array | ArrayLike | Scalar) -> Array:
        """
        Element-wise greater than or equal to comparison.

        Compares each value in the Array with the corresponding value in `other`.

        Args:
            other (Array | ArrayLike | Scalar): The other Array, ArrayLike, or Scalar to compare with.

        Returns:
            Array: A Array of boolean values indicating the result of the comparison.
        """
        other = cast(Array, other)
        return Array([s >= o for s, o in zip(self, other)])

    ###########################################################################
    # Operators
    ###########################################################################
    @_standardize_input
    def __add__(self, other: Array | ArrayLike | Scalar) -> Array:
        """
        Element-wise addition.

        Args:
            other (Array | ArrayLike | Scalar): The other Array, ArrayLike, or Scalar to operate with.

        Returns:
            Array: A Array with the results of the operation.
        """
        other = cast(Array, other)
        return Array([s + o for s, o in zip(self, other)])

    @_standardize_input
    def __sub__(self, other: Array | ArrayLike | Scalar) -> Array:
        """
        Element-wise subtraction.

        Args:
            other (Array | ArrayLike | Scalar): The other Array, ArrayLike, or Scalar to operate with.

        Returns:
            Array: A Array with the results of the operation.
        """
        other = cast(Array, other)
        return Array([s - o for s, o in zip(self, other)])

    @_standardize_input
    def __mul__(self, other: Array | ArrayLike | Scalar) -> Array:
        """
        Element-wise multiplication.

        Args:
            other (Array | ArrayLike | Scalar): The other Array, ArrayLike, or Scalar to operate with.

        Returns:
            Array: A Array with the results of the operation.
        """
        other = cast(Array, other)
        return Array([s * o for s, o in zip(self, other)])

    def __matmul__(self, other: Array | ArrayLike) -> Scalar:
        """
        Performs dot product with another Array, ArrayLike or Scalar.

        If other is a Array or a ArrayLike, performs the dot product between the two.

        Args:
            other (Array | ArrayLike)

        Returns:
            Scalar: The dot product of the Array.
        """
        other = cast(Array, other)
        return self.dot(other)

    @_standardize_input
    def __truediv__(self, other: Array | ArrayLike | Scalar) -> Array:
        """
        Element-wise division.

        Args:
            other (Array | ArrayLike | Scalar): The other Array, ArrayLike, or Scalar to operate with.

        Returns:
            Array: A Array with the results of the operation.
        """
        other = cast(Array, other)
        return Array([s / o for s, o in zip(self, other)])

    @_standardize_input
    def __floordiv__(self, other: Array | ArrayLike | Scalar) -> Array:
        """
        Element-wise floor division.

        Args:
            other (Array | ArrayLike | Scalar): The other Array, ArrayLike, or Scalar to operate with.

        Returns:
            Array: A Array with the results of the operation.
        """
        other = cast(Array, other)
        return Array([s // o for s, o in zip(self, other)])

    @_standardize_input
    def __mod__(self, other: Array | ArrayLike | Scalar) -> Array:
        """
        Element-wise modulo.

        Args:
            other (Array | ArrayLike | Scalar): The other Array, ArrayLike, or Scalar to operate with.

        Returns:
            Array: A Array with the results of the operation.
        """
        other = cast(Array, other)
        return Array([s % o for s, o in zip(self, other)])

    @_standardize_input
    def __divmod__(self, other: Array | ArrayLike | Scalar) -> Array:
        """
        Element-wise divmod.

        Args:
            other (Array | ArrayLike | Scalar): The other Array, ArrayLike, or Scalar to operate with.

        Returns:
            Array: A Array with the results of the operation.
        """
        other = cast(Array, other)
        return Array([divmod(s, o) for s, o in zip(self, other)])

    @_standardize_input
    def __pow__(self, other: Array | ArrayLike | Scalar) -> Array:
        """
        Element-wise exponentiation.

        Args:
            other (Array | ArrayLike | Scalar): The other Array, ArrayLike, or Scalar to operate with.

        Returns:
            Array: A Array with the results of the operation.
        """
        other = cast(Array, other)
        return Array([pow(s, o) for s, o in zip(self, other)])

    @_standardize_input
    def __lshift__(self, other: Array | ArrayLike | Scalar) -> Array:
        """
        Element-wise left bit shift.

        Args:
            other (Array | ArrayLike | Scalar): The other Array, ArrayLike, or Scalar to operate with.

        Returns:
            Array: A Array with the results of the operation.
        """
        other = cast(Array, other)
        return Array([s << o for s, o in zip(self, other)])

    @_standardize_input
    def __rshift__(self, other: Array | ArrayLike | Scalar) -> Array:
        """
        Element-wise right bit shift.

        Args:
            other (Array | ArrayLike | Scalar): The other Array, ArrayLike, or Scalar to operate with.

        Returns:
            Array: A Array with the results of the operation.
        """
        other = cast(Array, other)
        return Array([s >> o for s, o in zip(self, other)])

    @_standardize_input
    def __and__(self, other: Array | ArrayLike | Scalar) -> Array:
        """
        Element-wise AND.

        Args:
            other (Array | ArrayLike | Scalar): The other Array, ArrayLike, or Scalar to operate with.

        Returns:
            Array: A Array with the results of the operation.
        """
        other = cast(Array, other)
        return Array([s & o for s, o in zip(self, other)])

    @_standardize_input
    def __xor__(self, other: Array | ArrayLike | Scalar) -> Array:
        """
        Element-wise XOR.

        Args:
            other (Array | ArrayLike | Scalar): The other Array, ArrayLike, or Scalar to operate with.

        Returns:
            Array: A Array with the results of the operation.
        """
        other = cast(Array, other)
        return Array([s ^ o for s, o in zip(self, other)])

    @_standardize_input
    def __or__(self, other: Array | ArrayLike | Scalar) -> Array:
        """
        Element-wise OR.

        Args:
            other (Array | ArrayLike | Scalar): The other Array, ArrayLike, or Scalar to operate with.

        Returns:
            Array: A Array with the results of the operation.
        """
        other = cast(Array, other)
        return Array([s | o for s, o in zip(self, other)])

    ###########################################################################
    # Right-hand Side Operators
    ###########################################################################
    @_standardize_input
    def __radd__(self, other: Array | ArrayLike | Scalar) -> Array:
        return other + self

    @_standardize_input
    def __rsub__(self, other: Array | ArrayLike | Scalar) -> Array:
        return other - self

    @_standardize_input
    def __rmul__(self, other: Array | ArrayLike | Scalar) -> Array:
        return other * self

    @_standardize_input
    def __rmatmul__(self, other: Array | ArrayLike) -> Scalar:
        return other @ self

    @_standardize_input
    def __rtruediv__(self, other: Array | ArrayLike | Scalar) -> Array:
        return other / self

    @_standardize_input
    def __rfloordiv__(self, other: Array | ArrayLike | Scalar) -> Array:
        return other // self

    @_standardize_input
    def __rmod__(self, other: Array | ArrayLike | Scalar) -> Array:
        other = cast(Array, other)
        return other % self

    @_standardize_input
    def __rdivmod__(self, other: Array | ArrayLike | Scalar) -> Array:
        return divmod(other, self)

    @_standardize_input
    def __rpow__(self, other: Array | ArrayLike | Scalar) -> Array:
        return pow(other, self)  # type: ignore

    @_standardize_input
    def __rlshift__(self, other: Array | ArrayLike | Scalar) -> Array:
        return other << self

    @_standardize_input
    def __rrshift__(self, other: Array | ArrayLike | Scalar) -> Array:
        return other >> self

    @_standardize_input
    def __rand__(self, other: Array | ArrayLike | Scalar) -> Array:
        return other & self

    @_standardize_input
    def __rxor__(self, other: Array | ArrayLike | Scalar) -> Array:
        return other ^ self

    @_standardize_input
    def __ror__(self, other: Array | ArrayLike | Scalar) -> Array:
        return other | self

    ###########################################################################
    # In-place Operators
    ###########################################################################
    @_standardize_input
    def __iadd__(self, other: Array | ArrayLike | Scalar) -> Self:
        other = cast(Array, other)
        for i, o in enumerate(other):
            self[i] += o
        return self

    @_standardize_input
    def __isub__(self, other: Array | ArrayLike | Scalar) -> Self:
        other = cast(Array, other)
        for i, o in enumerate(other):
            self[i] -= o
        return self

    @_standardize_input
    def __imul__(self, other: Array | ArrayLike | Scalar) -> Self:
        other = cast(Array, other)
        for i, o in enumerate(other):
            self[i] *= o
        return self

    @_standardize_input
    def __imatmul__(self, other: Array | ArrayLike) -> Scalar:  # noqa: PYI034
        return self.dot(other)

    @_standardize_input
    def __itruediv__(self, other: Array | ArrayLike | Scalar) -> Self:
        other = cast(Array, other)
        for i, o in enumerate(other):
            self[i] /= o
        return self

    @_standardize_input
    def __ifloordiv__(self, other: Array | ArrayLike | Scalar) -> Self:
        other = cast(Array, other)
        for i, o in enumerate(other):
            self[i] //= o
        return self

    @_standardize_input
    def __imod__(self, other: Array | ArrayLike | Scalar) -> Self:
        other = cast(Array, other)
        for i, o in enumerate(other):
            self[i] %= o
        return self

    @_standardize_input
    def __ipow__(self, other: Array | ArrayLike | Scalar) -> Self:
        other = cast(Array, other)
        for i, o in enumerate(other):
            self[i] **= o
        return self

    @_standardize_input
    def __ilshift__(self, other: Array | ArrayLike | Scalar) -> Self:
        other = cast(Array, other)
        for i, o in enumerate(other):
            self[i] <<= o
        return self

    @_standardize_input
    def __irshift__(self, other: Array | ArrayLike | Scalar) -> Self:
        other = cast(Array, other)
        for i, o in enumerate(other):
            self[i] >>= o
        return self

    @_standardize_input
    def __iand__(self, other: Array | ArrayLike | Scalar) -> Self:
        other = cast(Array, other)
        for i, o in enumerate(other):
            self[i] &= o
        return self

    @_standardize_input
    def __ixor__(self, other: Array | ArrayLike | Scalar) -> Self:
        other = cast(Array, other)
        for i, o in enumerate(other):
            self[i] ^= o
        return self

    @_standardize_input
    def __ior__(self, other: Array | ArrayLike | Scalar) -> Self:  # type: ignore
        other = cast(Array, other)
        for i, o in enumerate(other):
            self[i] |= o
        return self

    ###########################################################################
    # Unary Operators
    ###########################################################################
    def __neg__(self) -> Array:
        return Array([-v for v in self])

    def __pos__(self) -> Array:
        return Array([+v for v in self])

    def __abs__(self) -> Array:
        return self.abs()

    def __invert__(self) -> Array:
        return Array([~v for v in self])


###########################################################################
# Functions
###########################################################################
# def merge(
#         left: DataFrame,
#         right: DataFrame,
#         how:DfMergeHow="inner",
#         on: Scalar | IndexLike | None = None,
#         *,
#         left_on: Scalar | IndexLike | None = None,
#         right_on: Scalar | IndexLike | None = None,
#         left_index: bool = False,
#         right_index: bool = False,
#         suffixes: tuple[str, str]=("_x", "_y")
#     ) -> DataFrame:
#     if how not in ('inner', 'left', 'right', 'outer'):
#         raise ValueError(f"how must be 'inner', 'left', 'right' or 'outer', got {how}")
#     left_on, right_on = _validate_merge_keys(on, left_on, right_on)
#     return DataFrame()

# def _validate_merge_keys(on: Scalar | IndexLike | None = None,
#         left_on: Scalar | IndexLike | None = None,
#         right_on: Scalar | IndexLike | None = None,
#         left_index: bool = False,
#         right_index: bool = False,
#         ) -> tuple[IndexLike, IndexLike]:
#     match on, left_on, right_on, left_index, right_index:
#         case s, None, None, False, False if _is_scalar(s):
#             left_on, right_on = [on], [on]
#         case s, None, None, False, False if _is_array_like(s):
#             left_on, right_on = on, on
#         case None, sl, sr, False, False if _is_scalar(sl) and _is_scalar(sr):
#             left_on, right_on = [left_on], [right_on]
#         case None, sl, sr, False, False if _is_array_like(sl) and _is_array_like(sr):
#             pass
#         case _:
#             msg = "Merge requires either an 'on' argument or 'left_on' and 'right_on'. Not both nor neither."
#             ValueError(msg)
#     return left_on, right_on


###########################################################################
# Indexers
###########################################################################
class Index(Array):
    name: Scalar | None
    _rev_index: dict[Scalar, list[int]]
    __slots__ = ("name", "_rev_index")

    def __init__(self, data: Index | ArrayLike | Iterator, name: Scalar | None = None):
        self.name = name
        if isinstance(data, Index):
            super().__init__(data.data)
            if name is None:
                self.name = data.name
        else:
            super().__init__(data)

        _rev_index = defaultdict(list)
        for i, d in enumerate(self.data):
            _rev_index[d].append(i)
        self._rev_index = dict(_rev_index)

    def __repr__(self) -> str:
        match self.name:
            case None:
                return f"{self.__class__.__name__}({self.data})"
            case name:
                return f'{self.__class__.__name__}({self.data}, name="{name!s}")'

    def __setitem__(self, *_args, **_kwargs):
        msg = "Index does not support mutable operations"
        raise TypeError(msg)

    def __delitem__(self, *_args, **_kwargs):
        msg = "Index does not support mutable operations"
        raise TypeError(msg)

    @property
    def values(self) -> list[Any]:
        """
        Return a list representation of the Index.

        Returns:
            list: The values of the Index.
        """
        return self.data

    def get_ilocs(self, key: LocIndexes) -> int | list[int]:
        match key:
            case Series():
                if _is_boolean_mask(key.values):
                    return [i for i, v in enumerate(key) if v]
                return [index for label in key.values for index in self._rev_index[label]]
            case Array() | list():
                if _is_boolean_mask(key):
                    return [i for i, v in enumerate(key) if v]
                return [index for label in key for index in self._rev_index[label]]
            case slice():
                return list(range(len(self)))[key]
            case k if _is_scalar(k):
                match self._rev_index[key]:
                    case [i]:
                        return i
                    case m:
                        return m
            case _:
                msg = f"Cannot index with: {key=}"
                raise KeyError(msg)


class BaseIndexer(Generic[T]):
    frame: T

    def __init__(self, data: T):
        self.frame = data


class LocSeriesIndexer(BaseIndexer["Series"]):
    def __getitem__(self, key: LocIndexes) -> Scalar | Series:
        return self.frame.iloc[self.frame.index.get_ilocs(key)]

    def __setitem__(self, key: LocIndexes, value: Scalar | ArrayLike | Mapping | Series):
        if _is_scalar(key) and key not in self.frame.index:
            self.frame._inplace_append({key: value})  # noqa: SLF001
        else:
            self.frame.iloc[self.frame.index.get_ilocs(key)] = value

    def __delitem__(self, key: LocIndexes):
        idxs = self.frame.index.get_ilocs(key)
        ilocs: list[int] = [idxs] if isinstance(idxs, int) else idxs
        new_data = list(self.frame.values)
        new_index = list(self.frame.index)
        for iloc in sorted(ilocs, reverse=True):
            del new_data[iloc]
            del new_index[iloc]
        self.frame._copy_from(Series(new_data, index=new_index, name=self.frame.name))  # noqa: SLF001


class IlocSeriesIndexer(BaseIndexer["Series"]):
    @overload
    def __getitem__(self, key: int) -> Scalar: ...  # no cov
    @overload
    def __getitem__(self, key: list[int] | slice | Series) -> Series: ...  # no cov
    def __getitem__(self, key: IlocIndexes) -> Scalar | Series:
        if isinstance(key, Series):
            key = key.values
        match key:
            case Array() | list() | slice():
                return Series(self.frame.values[key], index=self.frame.index[key])
            case k if isinstance(k, int):
                return self.frame.values[key]
            case _:
                msg = f"Cannot index with: {key=}"
                raise KeyError(msg)

    def __setitem__(self, key: IlocIndexes, value: Scalar | ArrayLike | Mapping | Series):
        if isinstance(key, Series):
            key = key.values
        match value:
            case Series():
                value = value.values
            case Mapping():
                value = list(value.values())

        match key:
            case Array() | list() | slice() as k:
                self.frame.values[k] = value
            case k if _is_scalar(k):
                self.frame.values[k] = value
            case _:
                msg = f"Cannot index with: {key=}"
                raise KeyError(msg)


class LocDataFrameIndexer(BaseIndexer["DataFrame"]):
    def __getitem__(self, key: LocIndexes | tuple[LocIndexes, LocIndexes]) -> Series | DataFrame:
        match key:
            case (row_indexer, col_indexer):
                return self.frame.iloc[
                    self.frame.index.get_ilocs(row_indexer), self.frame.columns.get_ilocs(col_indexer)
                ]
            case (_, *_):
                msg = "Onooooo"
                raise ValueError(msg)
            case indexer:
                return self.frame.iloc[self.frame.index.get_ilocs(indexer)]


class IlocDataFrameIndexer(BaseIndexer["DataFrame"]):
    @overload
    def __getitem__(self, key: tuple[int, int]) -> Scalar: ...  # no cov
    @overload
    def __getitem__(self, key: int | tuple[int, Any] | tuple[Any, int]) -> Series: ...  # no cov
    @overload
    def __getitem__(self, key: Any) -> DataFrame: ...  # no cov
    def __getitem__(self, key: IlocIndexes | tuple[IlocIndexes, IlocIndexes]) -> LocDataFrameReturn:
        row_indexer: IlocIndexes
        col_indexer: IlocIndexes
        match key:
            case (ri, ci):
                row_indexer, col_indexer = ri, ci
            case (_, *_):
                msg = "Onooooo"
                raise ValueError(msg)
            case _:
                row_indexer = key
                col_indexer = slice(None, None, None)

        if isinstance(row_indexer, Series):
            row_indexer = row_indexer.values
        if isinstance(col_indexer, Series):
            col_indexer = col_indexer.values

        match (row_indexer, col_indexer):
            case (Array() | list() | slice(), Array() | list() | slice()):
                return DataFrame(
                    self.frame.values[row_indexer][col_indexer],
                    index=self.frame.index[row_indexer],
                    columns=self.frame.columns[col_indexer],
                )
            case (r, Array() | list() | slice()) if isinstance(r, int):
                rows = self.frame.values[r]
                if len(rows) == 0:
                    return Series([], name=self.frame.index[r])
                return Series(rows[col_indexer], index=self.frame.columns[col_indexer], name=self.frame.index[r])
            case (Array() | list() | slice(), c) if isinstance(c, int):
                rows = self.frame.values[row_indexer]
                if len(rows) == 0:
                    return Series([], name=self.frame.columns[c])
                return Series(rows[row_indexer], index=self.frame.index[row_indexer], name=self.frame.columns[c])
            case (r, c) if isinstance(r, int) and isinstance(c, int):
                return self.frame.values[r][c]
            case _:
                msg = f"Cannot index with: {row_indexer=}, {col_indexer=}"
                raise KeyError(msg)


###########################################################################
# Series
###########################################################################
class Series(Sequence, Sized):
    """
    Series class representing a one-dimensional labeled array with capabilities for data analysis.

    Attributes:
        name (Scalar): Name of the Series.
    """

    name: Scalar | None
    _data: Array
    _index: Index
    loc: LocSeriesIndexer
    iloc: IlocSeriesIndexer
    __slots__ = ("name", "_data", "_index", "loc", "iloc")

    ###########################################################################
    # Initializer and general methods
    ###########################################################################
    def __init__(
        self,
        data: Series | Mapping | ArrayLike | Scalar | None = None,
        index: Index | IndexLike | None = None,
        name: Scalar | None = None,
    ):
        """
        Initializes a Series object.

        Args:
            data (Mapping | ArrayLike | Scalar, optional): Data for the Series. Can be a dictionary, list, or scalar. Defaults to None.
            index (Index | IndexLike, optional): IndexLike for the Series. Defaults to None.
            name (Scalar, optional): Name to assign to the Series. Defaults to None.

        Raises:
            ValueError: If the length of data and index don't match, or if data type is unexpected.
        """
        if isinstance(data, Series):
            self._copy_from(data)
            return
        self.name = name
        array_index, array_data = self._normalize_data(data)
        self._data = array_data
        self._index = self._validate_index(array_index, index)
        self._set_indexers()

    @staticmethod
    def _normalize_data(data: Mapping | ArrayLike | Scalar | None) -> tuple[IndexLike, Array]:
        match data:
            case None:
                return [], Array([])
            case Mapping():
                return list(data.keys()), Array(data.values())
            case ArrayLike():
                return list(range(len(data))), Array(data)
            case d if _is_scalar(d):
                return [0], Array([data])
            case _:
                msg = f"Unexpected data type: {type(data)=}"
                raise ValueError(msg)

    @staticmethod
    def _validate_index(array_index: IndexLike, index: Index | IndexLike | None = None) -> Index:
        match index:
            case None:
                return Index(array_index)
            case _:
                if len(array_index) != len(list(index)):
                    msg = f"Length of values ({len(array_index)}) does not match length of index ({len(list(index))})"
                    raise ValueError(msg)
                return Index(index)

    def _set_indexers(self):
        self.loc = LocSeriesIndexer(self)
        self.iloc = IlocSeriesIndexer(self)

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator:
        return iter(self.values)

    def __repr__(self) -> str:
        match self.values, self.name:
            case [[], None]:
                return f"{self.__class__.__name__}([])"
            case [[], name]:
                return f'{self.__class__.__name__}([], name="{name!s}")'
            case _:
                return self._repr_not_empty()

    def _repr_not_empty(self) -> str:
        columns = [self.index, self.values]
        widths = [max([len(str(v)) for v in col]) for col in columns]
        height = len(columns[0])
        ret = [[f"{col[i]!s:>{width}}" for col, width in zip(columns, widths)] for i in range(height)]
        return "\n".join("  ".join(r) for r in ret) + f"\nname: {self.name!s}\n"

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

    def rename(self, name: Scalar) -> Series:
        """
        Renames the Series.

        Args:
            name (Scalar): The new name for the Series.

        Returns:
            Series: A new Series with the updated name (a copy).
        """
        clone = self.copy(deep=True)
        clone.name = name
        return clone

    @property  # type: ignore
    def index(self) -> Index:
        """
        Returns the index of the Series.

        Returns:
            IndexLike: The index of the Series.
        """
        return self._index

    @index.setter
    def index(self, index: Index | IndexLike | Iterator):
        """
        Sets the index of the Series.

        Args:
            value (IndexLike): The new index for the Series.

        Raises:
            ValueError: If the length of the new index does not match the length of the Series.
        """
        if isinstance(index, Iterator):
            index = list(index)
        if len(self) != len(index):
            msg = f"Length mismatch: Expected axis has {len(self)} elements, new values have {len(index)} elements"
            raise ValueError(msg)
        self._index = Index(index)

    def reindex(self, index: Index | IndexLike | Iterator) -> Series:
        """
        Sets the index of the Series.

        Args:
            value (IndexLike): The new index for the Series.

        Raises:
            ValueError: If the length of the new index does not match the length of the Series.
        """
        clone = self.copy(deep=True)
        clone.index = index
        return clone

    def drop(self, indexes: LocIndexes) -> Series:
        """
        Removes the elements associated to `labels` and returns a new Series.

        Args:
            indexes (LocIndexes): Labels to remove

        Returns:
            Series: A new series with the removed values
        """
        clone = self.copy(deep=True)
        del clone[indexes]
        return clone

    @property
    def values(self) -> Array:  # type: ignore
        """
        Return a list representation of the Series.

        Returns:
            list: The values of the Series.
        """
        return self._data

    @property
    def shape(self) -> tuple[int,]:
        """
        Return a tuple of the shape of the underlying data.
        """
        return (len(self.index),)

    ###########################################################################
    # Accessors
    ###########################################################################
    def __getitem__(self, key: LocIndexes) -> Any | Series:
        """
        Retrieves an item or slice from the Series.

        Args:
            key (Scalar | list[Scalar] | slice | Series): The key, list of keys, or slice to retrieve.

        Returns:
            Any | Series: The value(s) associated with the given key(s) or slice.
        """
        return self.loc[key]

    def __setitem__(self, key: LocIndexes, value: Scalar | ArrayLike | Mapping | Series):
        """
        Sets an item or slice from the Series to a given value.

        Args:
            key (LocIndexes): The key, list of keys, or slice to retrieve.
            value (Scalar | ArrayLike | Mapping | Series): The value or values to set
        """
        self.loc[key] = value

    def __delitem__(self, key: LocIndexes):
        """
        Deletes an item or slice from the Series

        Args:
            key (LocIndexes): The key, list of keys, or slice to delete.
        """
        del self.loc[key]

    def head(self, n: int = 5) -> Series:
        """
        Returns the first n rows.

        Args:
            n (int, optional): Number of rows to return. Defaults to 5.

        Returns:
            Series: A new Series containing the first n rows.
        """
        return self[:n]

    def tail(self, n: int = 5) -> Series:
        """
        Returns the last n rows.

        Args:
            n (int, optional): Number of rows to return. Defaults to 5.

        Returns:
            Series: A new Series containing the last n rows.
        """
        return self[-n:]

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

    def find(self, val: Any) -> Scalar | None:
        """
        Finds the first label (key) associated with a given value in the Series.

        Args:
            val (Any): The value to search for.

        Returns:
            Scalar | None: The label (key) of the first occurrence of the value,
                             or None if the value is not found.
        """
        for k, v in self.iteritems():
            if v == val:
                return k
        return None

    def iteritems(self) -> Generator[tuple[Scalar, Scalar]]:
        yield from zip(self._index, self._data)

    ###########################################################################
    # Merge/Concatenate
    ###########################################################################
    def _copy_from(self, other: Series):
        self._data = other.values.copy()
        self._index = other.index.copy()
        self._set_indexers()

    def _inplace_append(self, other: Series | Mapping):
        new_series: Series
        match other:
            case Series():
                duplicate_index = set(self.index) & set(other.index)
                if len(duplicate_index) > 0:
                    msg = f"Cannot append with duplicate indexes: {duplicate_index}"
                    raise ValueError(msg)
                new_series = Series(self.to_dict() | other.to_dict(), name=self.name)
            case Mapping():
                duplicate_index = set(self.index) & set(other.keys())
                if len(duplicate_index) > 0:
                    msg = f"Cannot append with duplicate indexes: {duplicate_index}"
                    raise ValueError(msg)
                new_series = Series(self.to_dict() | dict(other), name=self.name)
            case _:
                msg = f"Cannot append with: {other=}"
                raise ValueError(msg)
        self._copy_from(new_series)

    def append(self, other: Series | Mapping) -> Series:
        """
        Appends `other` to the end of the Series

        Args:
            other (Series | Mapping): The data to append

        Returns:
            Series: A new Series with new data
        """
        # @TODO: Should we support ArrayLike too?
        clone = self.copy(deep=True)
        clone._inplace_append(other)  # noqa: SLF001
        return clone

    ###########################################################################
    # Auxiliary Functions
    ###########################################################################
    def _index_matches(self, index: Index):
        if set(self.index) != set(index):
            msg = "Indexes do not match"
            raise ValueError(msg)

    @staticmethod
    def _standardize_input(method: Callable) -> Callable:
        @functools.wraps(method)
        def wrapper(self: Series, other: Series | ArrayLike | Scalar) -> Any:
            match other:
                case Series():
                    self._validate_length(other)
                    self._index_matches(other.index)
                case o if _is_array_like(o):
                    other = Series(o, index=self.index, name=self.name)
                    self._validate_length(other)
                case _ as o:
                    other = Series(Array.full(len(self), o), index=self.index, name=self.name)  # type: ignore  # We let any type in
            return method(self, other)

        return wrapper

    def _validate_length(self, other: Series | ArrayLike):
        if len(self) != len(other):
            msg = f"Cannot operate Series with different sizes: {len(self)=}, {len(other)=}"
            raise ValueError(msg)

    ###########################################################################
    # Map/Reduce
    ###########################################################################
    def map(self, func: Callable[[Scalar], Scalar]) -> Series:
        """
        Applies a function to each value in the Series.

        Args:
            func (Callable[[Scalar], Scalar]): The function to apply.

        Returns:
            Series: A new Series with the results of the function applied.
        """
        return Series(self._data.map(func), index=self.index, name=self.name)

    def reduce(self, func: Callable[[Any, tuple[Scalar, Scalar]], Any], initial: Any) -> Any:
        """
        Reduces the Series using a function.

        Args:
            func (Callable[[Any, tuple[Scalar, Scalar]], Any]): The function to apply for reduction.
            initial (Any): The initial value for the reduction.

        Returns:
            Any: The reduced value.
        """
        if len(self) > 0:
            return reduce(func, self.iteritems(), initial)
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
        return func(self._data)

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
        return self.map(abs)  # type: ignore

    @_standardize_input
    def dot(self, other: Series | ArrayLike) -> Scalar:
        """
        Performs dot product with another Series, ArrayLike or Scalar.

        If other is a Series or a ArrayLike, performs the dot product between the two.
        If other is a Scalar, multiplies all elements of the Series by the scalar and returns the sum.

        Args:
            other (Series | ArrayLike | Scalar)

        Returns:
            Scalar: The dot product of the Series.
        """
        other = cast(Series, other)
        return sum(other[k] * v for k, v in self.iteritems())

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
        Returns True if all values in the Series are truthy.

        Returns:
            bool: True if all values are truthy, False otherwise.
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

    def idxmax(self) -> Scalar | None:
        """
        Returns the label of the maximum value.

        Returns:
            Scalar: The label of the maximum value.
        """
        if len(self) == 0:
            msg = "Attempt to get ixmax of an empty sequence"
            raise ValueError(msg)
        return self.find(self.max())

    def idxmin(self) -> Scalar | None:
        """
        Returns the label of the minimum value.

        Returns:
            Scalar: The label of the minimum value.
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

    def quantiles(self, *, n=4, method: Literal["exclusive", "inclusive"] = "exclusive") -> ArrayLike[float]:
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
        return self._data.to_list()

    def to_dict(self) -> dict[Scalar, Any]:
        """
        Converts the Series to a dictionary.

        Returns:
            dict[Scalar, Any]: A dictionary representation of the Series.
        """
        return dict(zip(self.index, self._data))

    ###########################################################################
    # Comparisons
    ###########################################################################
    @_standardize_input
    def __lt__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise less than comparison.

        Compares each value in the Series with the corresponding value in `other`.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to compare with.

        Returns:
            Series: A Series of boolean values indicating the result of the comparison.
        """
        other = cast(Series, other)
        return Series({k: v < other[k] for k, v in self.iteritems()}, name=self.name)

    @_standardize_input
    def __le__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise less than or equal to comparison.

        Compares each value in the Series with the corresponding value in `other`.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to compare with.

        Returns:
            Series: A Series of boolean values indicating the result of the comparison.
        """
        other = cast(Series, other)
        return Series({k: v <= other[k] for k, v in self.iteritems()}, name=self.name)

    @_standardize_input
    def __eq__(self, other: Series | ArrayLike | Scalar) -> Series:  # type: ignore
        """
        Element-wise equality comparison.

        Compares each value in the Series with the corresponding value in `other`.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to compare with.

        Returns:
            Series: A Series of boolean values indicating the result of the comparison.
        """
        other = cast(Series, other)
        return Series({k: v == other[k] for k, v in self.iteritems()}, name=self.name)

    @_standardize_input
    def __ne__(self, other: Series | ArrayLike | Scalar) -> Series:  # type: ignore
        """
        Element-wise inequality comparison.

        Compares each value in the Series with the corresponding value in `other`.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to compare with.

        Returns:
            Series: A Series of boolean values indicating the result of the comparison.
        """
        other = cast(Series, other)
        return Series({k: v != other[k] for k, v in self.iteritems()}, name=self.name)

    @_standardize_input
    def __gt__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise greater than comparison.

        Compares each value in the Series with the corresponding value in `other`.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to compare with.

        Returns:
            Series: A Series of boolean values indicating the result of the comparison.
        """
        other = cast(Series, other)
        return Series({k: v > other[k] for k, v in self.iteritems()}, name=self.name)

    @_standardize_input
    def __ge__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise greater than or equal to comparison.

        Compares each value in the Series with the corresponding value in `other`.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to compare with.

        Returns:
            Series: A Series of boolean values indicating the result of the comparison.
        """
        other = cast(Series, other)
        return Series({k: v >= other[k] for k, v in self.iteritems()}, name=self.name)

    ###########################################################################
    # Operators
    ###########################################################################
    @_standardize_input
    def __add__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise addition.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = cast(Series, other)
        return Series({k: v + other[k] for k, v in self.iteritems()}, name=self.name)

    @_standardize_input
    def __sub__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise subtraction.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = cast(Series, other)
        return Series({k: v - other[k] for k, v in self.iteritems()}, name=self.name)

    @_standardize_input
    def __mul__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise multiplication.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = cast(Series, other)
        return Series({k: v * other[k] for k, v in self.iteritems()}, name=self.name)

    @_standardize_input
    def __matmul__(self, other: Series | ArrayLike) -> Scalar:
        """
        Performs dot product with another Series, ArrayLike or Scalar.

        If other is a Series or a ArrayLike, performs the dot product between the two.

        Args:
            other (Series | ArrayLike)

        Returns:
            Scalar: The dot product of the Series.
        """
        other = cast(Series, other)
        return self.dot(other)

    @_standardize_input
    def __truediv__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise division.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = cast(Series, other)
        return Series({k: v / other[k] for k, v in self.iteritems()}, name=self.name)

    @_standardize_input
    def __floordiv__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise floor division.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = cast(Series, other)
        return Series({k: v // other[k] for k, v in self.iteritems()}, name=self.name)

    @_standardize_input
    def __mod__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise modulo.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = cast(Series, other)
        return Series({k: v % other[k] for k, v in self.iteritems()}, name=self.name)

    @_standardize_input
    def __divmod__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise divmod.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = cast(Series, other)
        return Series({k: divmod(v, other[k]) for k, v in self.iteritems()}, name=self.name)

    @_standardize_input
    def __pow__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise exponentiation.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = cast(Series, other)
        return Series({k: pow(v, other[k]) for k, v in self.iteritems()}, name=self.name)  # type: ignore

    @_standardize_input
    def __lshift__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise left bit shift.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = cast(Series, other)
        return Series({k: v << other[k] for k, v in self.iteritems()}, name=self.name)

    @_standardize_input
    def __rshift__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise right bit shift.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = cast(Series, other)
        return Series({k: v >> other[k] for k, v in self.iteritems()}, name=self.name)

    @_standardize_input
    def __and__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise AND.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = cast(Series, other)
        return Series({k: v & other[k] for k, v in self.iteritems()}, name=self.name)

    @_standardize_input
    def __xor__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise XOR.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = cast(Series, other)
        return Series({k: v ^ other[k] for k, v in self.iteritems()}, name=self.name)

    @_standardize_input
    def __or__(self, other: Series | ArrayLike | Scalar) -> Series:
        """
        Element-wise OR.

        Args:
            other (Series | ArrayLike | Scalar): The other Series, ArrayLike, or Scalar to operate with.

        Returns:
            Series: A Series with the results of the operation.
        """
        other = cast(Series, other)
        return Series({k: v | other[k] for k, v in self.iteritems()}, name=self.name)

    ###########################################################################
    # Right-hand Side Operators
    ###########################################################################
    @_standardize_input
    def __radd__(self, other: Series | ArrayLike | Scalar) -> Series:
        return other + self

    @_standardize_input
    def __rsub__(self, other: Series | ArrayLike | Scalar) -> Series:
        return other - self

    @_standardize_input
    def __rmul__(self, other: Series | ArrayLike | Scalar) -> Series:
        return other * self

    def __rmatmul__(self, other: Series | ArrayLike) -> Scalar:
        return self.dot(other)

    @_standardize_input
    def __rtruediv__(self, other: Series | ArrayLike | Scalar) -> Series:
        return other / self

    @_standardize_input
    def __rfloordiv__(self, other: Series | ArrayLike | Scalar) -> Series:
        return other // self

    @_standardize_input
    def __rmod__(self, other: Series | ArrayLike | Scalar) -> Series:
        other = cast(Series, other)
        return other % self

    @_standardize_input
    def __rdivmod__(self, other: Series | ArrayLike | Scalar) -> Series:
        return divmod(other, self)

    @_standardize_input
    def __rpow__(self, other: Series | ArrayLike | Scalar) -> Series:
        return pow(other, self)  # type: ignore

    @_standardize_input
    def __rlshift__(self, other: Series | ArrayLike | Scalar) -> Series:
        return other << self

    @_standardize_input
    def __rrshift__(self, other: Series | ArrayLike | Scalar) -> Series:
        return other >> self

    @_standardize_input
    def __rand__(self, other: Series | ArrayLike | Scalar) -> Series:
        other = cast(Series, other)
        return other & self

    @_standardize_input
    def __rxor__(self, other: Series | ArrayLike | Scalar) -> Series:
        return other ^ self

    @_standardize_input
    def __ror__(self, other: Series | ArrayLike | Scalar) -> Series:
        return other | self

    ###########################################################################
    # In-place Operators
    ###########################################################################
    @_standardize_input
    def __iadd__(self, other: Series | ArrayLike | Scalar) -> Self:
        other = cast(Series, other)
        for k in self.index:
            self[k] += other[k]
        return self

    @_standardize_input
    def __isub__(self, other: Series | ArrayLike | Scalar) -> Self:
        other = cast(Series, other)
        for k in self.index:
            self[k] -= other[k]
        return self

    @_standardize_input
    def __imul__(self, other: Series | ArrayLike | Scalar) -> Self:
        other = cast(Series, other)
        for k in self.index:
            self[k] *= other[k]
        return self

    def __imatmul__(self, other: Series | ArrayLike) -> Scalar:  # type: ignore  # noqa: PYI034
        other = cast(Series, other)
        return self.dot(other)

    @_standardize_input
    def __itruediv__(self, other: Series | ArrayLike | Scalar) -> Self:
        other = cast(Series, other)
        for k in self.index:
            self[k] /= other[k]
        return self

    @_standardize_input
    def __ifloordiv__(self, other: Series | ArrayLike | Scalar) -> Self:
        other = cast(Series, other)
        for k in self.index:
            self[k] //= other[k]
        return self

    @_standardize_input
    def __imod__(self, other: Series | ArrayLike | Scalar) -> Self:
        other = cast(Series, other)
        for k in self.index:
            self[k] %= other[k]
        return self

    @_standardize_input
    def __ipow__(self, other: Series | ArrayLike | Scalar) -> Self:
        other = cast(Series, other)
        for k in self.index:
            self[k] **= other[k]
        return self

    @_standardize_input
    def __ilshift__(self, other: Series | ArrayLike | Scalar) -> Self:
        other = cast(Series, other)
        for k in self.index:
            self[k] <<= other[k]
        return self

    @_standardize_input
    def __irshift__(self, other: Series | ArrayLike | Scalar) -> Self:
        other = cast(Series, other)
        for k in self.index:
            self[k] >>= other[k]
        return self

    @_standardize_input
    def __iand__(self, other: Series | ArrayLike | Scalar) -> Self:
        other = cast(Series, other)
        for k in self.index:
            self[k] &= other[k]
        return self

    @_standardize_input
    def __ixor__(self, other: Series | ArrayLike | Scalar) -> Self:
        other = cast(Series, other)
        for k in self.index:
            self[k] ^= other[k]
        return self

    @_standardize_input
    def __ior__(self, other: Series | ArrayLike | Scalar) -> Self:  # type: ignore
        other = cast(Series, other)
        for k in self.index:
            self[k] |= other[k]
        return self

    ###########################################################################
    # Unary Operators
    ###########################################################################
    def __neg__(self) -> Series:
        return Series({k: -v for k, v in self.iteritems()})  # type: ignore

    def __pos__(self) -> Series:
        return Series({k: +v for k, v in self.iteritems()})  # type: ignore

    def __abs__(self) -> Series:
        return self.abs()

    def __invert__(self) -> Series:
        return Series({k: ~v for k, v in self.iteritems()})  # type: ignore


###########################################################################
# DataFrame
###########################################################################
class DataFrame:
    """
    DataFrame class representing a two-dimensional, size-mutable, tabular data structure with
    labeled axes (rows and columns).

    Attributes:
        index (IndexLike): The row labels (index) of the DataFrame. Used for label-based row selection.
        columns (IndexLike): The column labels of the DataFrame. Used for label-based column selection.
    """

    _index: Index
    _columns: Index
    _data: Array
    loc: LocDataFrameIndexer
    iloc: IlocDataFrameIndexer
    __slots__ = ("_index", "_columns", "_data", "loc", "iloc")

    ###########################################################################
    # Initializer and general methods
    ###########################################################################
    def __init__(
        self,
        data: Mapping[Scalar, Series] | Mapping[Scalar, ArrayLike] | ArrayLike | Iterator | None = None,
        index: IndexLike | None = None,
        columns: IndexLike | None = None,
    ):
        """
        Initializes a DataFrame object.

        Args:
            data (Mapping | ArrayLike | Scalar, optional): Data for the Series. Can be a dictionary, list, or scalar. Defaults to None.
            index (IndexLike, optional): IndexLike for the Series. Defaults to None.
            name (Scalar, optional): Name to assign to the Series. Defaults to None.

        Raises:
            ValueError: If the length of data and index don't match, or if data type is unexpected.
        """
        if isinstance(data, Iterator):
            data = list(data)

        match data:
            case None:
                self._init_empty(index, columns)
            case ArrayLike() | Mapping() if len(data) == 0:
                self._init_empty(index, columns)
            case Mapping() as m if all(isinstance(v, ArrayLike) for v in m.values()):
                self._init_mapping_of_series({col: Series(val) for col, val in data.items()}, index, columns)
            case ArrayLike() as c if all(isinstance(v, (Mapping, ArrayLike)) for v in c) and _is_array_like(c):
                self._init_collection_of_series([Series(row) for row in data], index, columns)
            case ArrayLike() as c if all(_is_scalar(v) for v in c) and _is_array_like(c):
                self._init_mapping_of_series({0: Series(data)}, index, columns)
            case _:
                msg = "DataFrame constructor not properly called!"
                raise ValueError(msg)
        self._validate_index_and_columns()
        self._set_indexers()

    def _init_empty(self, index: IndexLike | None = None, columns: IndexLike | None = None):
        if (index is not None and columns is not None) and (len(index) > 0 and len(columns) > 0):
            msg = "Cannot create an empty DataFrame with both columns and indexes. Please chose only one."
            raise ValueError(msg)
        self._index = Index([] if index is None else index)
        self._columns = Index([] if columns is None else columns)
        self._data = Array([Array([])])

    def _init_mapping_of_series(
        self, data: Mapping[Scalar, Series], index: IndexLike | None = None, columns: IndexLike | None = None
    ):
        col0 = next(iter(data))
        val0 = data[col0]
        if len(val0) == 0:
            columns = list(data.keys()) if columns is None else columns
            self._init_empty(index, columns)
            return

        self._index = Index(val0.index if index is None else index)
        self._columns = Index(list(data.keys()) if columns is None else columns)
        if (len(self._index) != len(val0.index)) or (len(self._columns) != len(data)):
            passed = (len(val0.index), len(data))
            implied = (len(self._index), len(self._columns))
            msg = f"Shape of passed values is {passed}, indices imply {implied}"
            raise ValueError(msg)
        self._data = Array([Array([data[c].iloc[i] for c in self.columns]) for i in range(len(self.index))])

    def _init_collection_of_series(
        self, data: list[Series], index: IndexLike | None = None, columns: IndexLike | None = None
    ):
        row0 = next(iter(data))
        src_columns = row0.index
        if len(row0) == 0:
            index = [0] if index is None else index
            self._init_empty(index, columns)
            return

        self._columns = Index(columns if columns is not None else src_columns)
        if any((d.index != src_columns).any() for d in data):
            all_cols = {item for d in data for item in d.index}
            missing_cols = all_cols - set(self._columns) or "{}"
            extra_cols = set(self._columns) - all_cols or "{}"
            msg = f"Misaligned columns. Expected {self._columns}. Missing: {missing_cols}, Extra: {extra_cols}"
            raise ValueError(msg)

        # # @TODO: Deal with Series with names
        self._index = Index(list(range(len(data))) if index is None else index)
        if (len(self._index) != len(data)) or (len(self._columns) != len(src_columns)):
            passed = (len(data), len(src_columns))
            implied = (len(self._index), len(self._columns))
            msg = f"Shape of passed values is {passed}, indices imply {implied}"
            raise ValueError(msg)

        self._data = Array([Array(row) for row in data])

    def _validate_index_and_columns(self):
        # @TODO: How to validate empty DataFrames?
        if len(self.columns) == 0:
            return
        if len(self.index) == 0:
            return

        if len(self._data) != len(self.index):
            msg = "Somehow indexes and DataFrame length don't match. This shouldn't happen!"
            raise ValueError(msg)

        for row in self._data:
            if len(row) != len(self.columns):
                msg = "Somehow columns and DataFrame width don't match. This shouldn't happen!"
                raise ValueError(msg)

    @staticmethod
    def _validate_axis(axis: Any):
        match axis:
            case int(c) if c in (AxisRows, AxisCols):
                pass
            case _:
                msg = f"No axis named {axis} for object type DataFrame"
                raise ValueError(msg)

    def _set_indexers(self):
        self.iloc = IlocDataFrameIndexer(self)
        self.loc = LocDataFrameIndexer(self)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.index), len(self.columns))

    def __len__(self) -> int:
        return len(self.index)

    def __repr__(self):
        if len(self) == 0 or len(self.iloc[0]) == 0:
            return f"Empty DataFrame\nColumns: {self.columns.values!s}\nIndex: {self.index.values!s}"
        columns = list(zip(["", *self.columns], *[[idx, *s] for idx, s in zip(self.index, self._data)]))
        widths = [max([len(str(v)) for v in col]) for col in columns]
        height = len(columns[0])
        ret = [[f"{col[i]!s:>{width}}" for col, width in zip(columns, widths)] for i in range(height)]
        return "\n".join("  ".join(r) for r in ret)

    def copy(self, *, deep: bool = True) -> DataFrame:
        """
        Creates a copy of the DataFrame.

        Args:
            deep (bool, optional): If True, creates a deep copy. Otherwise, creates a shallow copy. Defaults to True.

        Returns:
            DataFrame: A copy of the DataFrame.
        """
        clone = copy.deepcopy(self) if deep else copy.copy(self)
        clone._set_indexers()  # noqa: SLF001
        return clone

    @property
    def index(self) -> Index:
        """
        Returns the index of the DataFrame.

        Returns:
            IndexLike: The index of the DataFrame.
        """
        return self._index

    @index.setter
    def index(self, index: IndexLike):
        """
        Sets the index of the DataFrame.

        Args:
            value (IndexLike): The new index for the DataFrame.

        Raises:
            ValueError: If the length of the new index does not match the length of the Series.
        """
        if len(self) != len(index):
            msg = f"Length mismatch: Expected axis has {len(self)} elements, new values have {len(index)} elements"
            raise ValueError(msg)
        idx = Index(index)
        self._index = idx
        for s in self._data:
            s.index = idx

    @property
    def columns(self) -> Index:
        """
        Returns the columns of the DataFrame.

        Returns:
            IndexLike: The columns of the DataFrame.
        """
        return self._columns

    @columns.setter
    def columns(self, columns: IndexLike):
        """
        Sets the columns of the DataFrame.

        Args:
            columns (IndexLike): The new index for the DataFrame.

        Raises:
            ValueError: If the length of the new index does not match the length of the Series.
        """
        if len(self.columns) != len(columns):
            msg = f"Length mismatch: Expected axis has {len(self)} elements, new values have {len(columns)} elements"
            raise ValueError(msg)
        self._columns = Index(columns)

    @property
    def T(self) -> DataFrame:  # noqa: N802
        """
        Return the transpose of the DataFrame, switching rows and columns.

        The transpose (swap of axes) effectively converts columns to rows and rows to columns,
        maintaining the original data relationships while rotating the DataFrame structure.

        Returns:
            DataFrame: A new DataFrame instance where:
                - Original columns become the new index
                - Original index becomes the new columns
                - Data values are transposed accordingly
        """
        return DataFrame(dict(self.iterrows()), index=self.columns[:])

    @property
    def values(self) -> Array[Array[Any]]:  # type: ignore
        """
        Return a list representation of the DataFrame.

        Returns:
            list: The values of the DataFrame.
        """
        return self._data

    def iterrows(self) -> Generator[tuple[Scalar, Series]]:
        """
        Iterate over DataFrame rows as (index, Series) pairs.

        Provides an efficient way to loop through rows of the DataFrame, returning
        both the index label and the row data as a Series for each iteration.

        Returns:
            Generator[tuple[Scalar, Series]]: A generator yielding tuples containing:
                - Scalar: The label of the current row
                - Series: Row data
        """
        for idx, row in zip(self.index, self._data):
            yield idx, Series(row, index=self.columns, name=idx)

    ###########################################################################
    # Accessors
    ###########################################################################
    @overload
    def __getitem__(self, index: Scalar) -> Series: ...  # no cov
    @overload
    def __getitem__(self, index: Series | Array | list[Scalar] | slice) -> DataFrame: ...  # no cov
    def __getitem__(
        self, index: Series | Array | list[Scalar] | tuple[Scalar, Scalar] | slice | Scalar
    ) -> LocDataFrameReturn:
        """
        Retrieves an item or slice from the DataFrame.

        Args:
            key (Scalar | list[Scalar] | slice | Series | Array): The key, list of keys, or slice to retrieve.

        Returns:
            Series | DataFRame: A new Series if a single column is selected or a DataFrame for multiple columns
        """
        match index:
            case tuple():
                return self.loc[index]
            case Series() | Array() | list():
                if _is_boolean_mask(Series(index)):
                    return self.loc[index]
                return self.loc[:, index]
            case slice():
                return self.iloc[index]
            case k if _is_scalar(k):
                return self.loc[:, index]
            case _:
                msg = "oh no!"
                raise ValueError(msg)

    def head(self, n: int = 5) -> DataFrame:
        """
        Returns the first n rows.

        Args:
            n (int, optional): Number of rows to return. Defaults to 5.

        Returns:
            DataFrame: A new DataFrame containing the first n rows.
        """
        return self.iloc[:n]

    def tail(self, n: int = 5) -> DataFrame:
        """
        Returns the last n rows.

        Args:
            n (int, optional): Number of rows to return. Defaults to 5.

        Returns:
            DataFrame: A new DataFrame containing the last n rows.
        """
        return self.iloc[-n:]

    ###########################################################################
    # Apply/Agg/Map/Reduce
    ###########################################################################
    def apply(self, method: Callable[[Series], Any], axis: Axis = 0) -> Series:
        """
        Apply a function along a DataFrame axis (columns or rows).

        Processes either each column (axis=0) or each row (axis=1) as a Series object,
        applying the provided function to each Series. Returns a new Series with
        aggregated/transformed values.

        Args:
            method: Callable that takes a Series and returns a scalar value.
                - For axis=0: Receives each column as a Series
                - For axis=1: Receives each row as a Series
            axis: Axis along which to apply:
                - 0: Apply to each column (default)
                - 1: Apply to each row

        Returns:
            Series: Results of applying the method along specified axis.
        """
        self._validate_axis(axis)
        match axis:
            case int(c) if c == AxisRows:
                return Series({col: method(s) for col, s in self.iterrows()})
            case int(c) if c == AxisCols:
                return self.T.apply(method, axis=0)
            case unreachable:  # no cov
                assert_never(unreachable)  # type: ignore # @TODO: How to exhaust this check?

    def _apply_with_none(self, method: Callable[[Series], Any], axis: AxisOrNone = 0):
        match axis:
            case None:
                return method(self.apply(method, 0))
            case _:
                return self.apply(method, axis)

    #     def agg(self, method: Callable[[ArrayLike[Any]], Any], axis: Axis = 0) -> Series:
    #         """
    #         Aggregate data along specified axis using one or more operations.

    #         Applies aggregation function to raw array values (rather than Series objects)
    #         along columns (axis=0) or rows (axis=1). Optimized for numerical aggregations.

    #         Args:
    #             method: Callable that takes array-like data and returns scalar
    #                 - For axis=0: Receives column values as array
    #                 - For axis=1: Receives row values as array
    #             axis: Axis to aggregate along:
    #                 - 0: Aggregate each column (default)
    #                 - 1: Aggregate each row

    #         Returns:
    #             Series: Aggregation results.
    #         """
    #         self._validate_axis(axis)
    #         match axis:
    #             case int(c) if c == AxisRows:
    #                 return Series({col: method(s.values) for col, s in self.items()})
    #             case int(c) if c == AxisCols:
    #                 return self.T.agg(method, axis=0)
    #             case unreachable:  # no cov
    #                 assert_never(unreachable)  # type: ignore # @TODO: How to exhaust this check?

    #     def _agg_with_none(self, method: Callable[[ArrayLike[Any]], Any], axis: AxisOrNone = 0):
    #         match axis:
    #             case None:
    #                 return method([item for sublist in self.to_list() for item in sublist])
    #             case _:
    #                 return self.agg(method, axis)

    def map(self, func: Callable) -> DataFrame:
        """
        Applies a function to each value in the DataFrame.

        Args:
            func (Callable): The function to apply.

        Returns:
            DataFrame: A new DataFrame with the results of the function applied.
        """
        return DataFrame((s.map(func) for _, s in self.iterrows()), index=self.index, columns=self.columns)

    def astype(self, new_type: type) -> DataFrame:
        """
        Casts the DataFrame to a new type.

        Args:
            new_type (type): The type to cast to.

        Returns:
            DataFrame: A new DataFrame with the values cast to the new type.
        """
        return self.map(new_type)

    # @overload
    # def dot(self, other: DataFrame) -> DataFrame: ...  # no cov
    # @overload
    # def dot(self, other: Series | ArrayLike) -> Series: ...  # no cov
    def dot(self, other: DataFrame | Series | ArrayLike) -> DataFrame | Series:
        """
        Compute the matrix multiplication between the DataFrame and another DataFrame or Series.

        Args:
            other (DataFrame or Series): The other DataFrame or Series to multiply with.

        Returns:
            DataFrame or Series: The result of the matrix multiplication.

        Raises:
            ValueError: If the columns of the DataFrame do not match the index of the other DataFrame/Series.
            TypeError: If the other object is not a DataFrame or Series.
        """
        not_aligned_msg = "matrices are not aligned"
        match other:
            case DataFrame():
                if list(self.columns) != list(other.index):
                    raise ValueError(not_aligned_msg)
                df_data = [[s_a.dot(s_b) for s_b in other.T.values] for s_a in self.values]
                return DataFrame(df_data, index=self.index, columns=other.columns)
            case Series() | ArrayLike():
                if len(self.columns) != len(other):
                    raise ValueError(not_aligned_msg)
                if not isinstance(other, Series):
                    other = Series(other, index=self.columns)
                if list(self.columns) != list(other.index):
                    raise ValueError(not_aligned_msg)
                s_data = [sa.dot(other) for sa in self.values]
                return Series(s_data, index=self.index, name=other.name)
            case _:
                msg = "Dot product requires other to be a DataFrame or Series."
                raise TypeError(msg)

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
        """
        Returns the maximum value in the DataFrame.

        Args:
            axis: AxisOrNone to aggregate along:
                - 0: Aggregate each column (default)
                - 1: Aggregate each row
                - None: Aggregates along both axes returning a scalar

        Returns:
            Series | Scalar: The maximum values along the axis
        """
        return self._apply_with_none(lambda s: s.max(), axis)

    @overload
    def min(self) -> Series: ...  # no cov
    @overload
    def min(self, axis: Axis) -> Series: ...  # no cov
    @overload
    def min(self, axis: None) -> Scalar: ...  # no cov
    def min(self, axis: AxisOrNone = 0) -> Series | Scalar:
        """
        Returns the minimum value in the DataFrame.

        Args:
            axis: AxisOrNone to aggregate along:
                - 0: Aggregate each column (default)
                - 1: Aggregate each row
                - None: Aggregates along both axes returning a scalar

        Returns:
            Series | Scalar: The minimum values along the axis
        """
        return self._apply_with_none(lambda s: s.min(), axis)

    @overload
    def sum(self) -> Series: ...  # no cov
    @overload
    def sum(self, axis: Axis) -> Series: ...  # no cov
    @overload
    def sum(self, axis: None) -> Scalar: ...  # no cov
    def sum(self, axis: AxisOrNone = 0) -> Series | Scalar:
        """
        Returns the sum of the values in the DataFrame.

        Args:
            axis: AxisOrNone to aggregate along:
                - 0: Aggregate each column (default)
                - 1: Aggregate each row
                - None: Aggregates along both axes returning a scalar

        Returns:
            Series | Scalar: The sum of the values along the axis
        """
        return self._apply_with_none(lambda s: s.sum(), axis)

    @overload
    def all(self) -> Series: ...  # no cov
    @overload
    def all(self, axis: Axis) -> Series: ...  # no cov
    @overload
    def all(self, axis: None) -> bool: ...  # no cov
    def all(self, axis: AxisOrNone = 0) -> Series | bool:
        """
        Returns True if all values in the DataFrame are truthy.

        Args:
            axis: AxisOrNone to aggregate along:
                - 0: Aggregate each column (default)
                - 1: Aggregate each row
                - None: Aggregates along both axes returning a scalar

        Returns:
            Series | bool: True if all values are truthy, False otherwise.
        """
        return self._apply_with_none(lambda s: s.all(), axis)

    @overload
    def any(self) -> Series: ...  # no cov
    @overload
    def any(self, axis: Axis) -> Series: ...  # no cov
    @overload
    def any(self, axis: None) -> bool: ...  # no cov
    def any(self, axis: AxisOrNone = 0) -> Series | bool:
        """
        Returns True if any value in the DataFrame is truthy.

        Args:
            axis: AxisOrNone to aggregate along:
                - 0: Aggregate each column (default)
                - 1: Aggregate each row
                - None: Aggregates along both axes returning a scalar

        Returns:
            Series | bool: True if any value is truthy, False otherwise.
        """
        return self._apply_with_none(lambda s: s.any(), axis)

    def idxmax(self, axis: Axis = 0) -> Series:
        """
        Returns the labels of the maximum values.

        Args:
            axis: Axis to aggregate along:
                - 0: Aggregate each column (default)
                - 1: Aggregate each row

        Returns:
            Series: The labels of the maximum values
        """
        return self._apply_with_none(lambda s: s.idxmax(), axis)

    def idxmin(self, axis: Axis = 0) -> Series:
        """
        Returns the labels of the minimum values.

        Args:
            axis: Axis to aggregate along:
                - 0: Aggregate each column (default)
                - 1: Aggregate each row

        Returns:
            Series: The labels of the minimum values
        """
        return self._apply_with_none(lambda s: s.idxmin(), axis)

    ###########################################################################
    # GroupBy
    ###########################################################################

    #     ###########################################################################
    #     # Merge/Concatenate
    #     ###########################################################################
    #     def append(self, other: DataFrame, axis: Axis = 0) -> DataFrame:
    #         """
    #         Appends `other` to the end of the DataFrame

    #         Args:
    #             other (DataFrame): The data to append
    #             axis: Axis to aggregate along:
    #                 - 0: Aggregate each column (default)
    #                 - 1: Aggregate each row

    #         Returns:
    #             DataFrame: A new DataFrame with new data
    #         """
    #         # @TODO: Support more data types for other?
    #         self._validate_axis(axis)
    #         match axis:
    #             case int(c) if c == AxisRows:
    #                 missing_columns = set(self.columns) ^ set(other.columns)
    #                 if len(missing_columns) > 0:
    #                     msg = f"Cannot append data with missing columns: {missing_columns}"
    #                     raise ValueError(msg)
    #                 return DataFrame({col: s.append(other[col]) for col, s in self.data.items()})
    #             case int(c) if c == AxisCols:
    #                 missing_indexes = set(self.index) ^ set(other.index)
    #                 if len(missing_indexes) > 0:
    #                     msg = f"Cannot append data with missing indexes: {missing_indexes}"
    #                     raise ValueError(msg)
    #                 return DataFrame(dict(self.data | other.data))
    #             case unreachable:  # no cov
    #                 assert_never(unreachable)  # type: ignore # @TODO: How to exhaust this check?

    #     def merge(
    #         self,
    #         right: DataFrame,
    #         how:DfMergeHow="inner",
    #         on: Scalar | IndexLike | None = None,
    #         *,
    #         left_on: Scalar | IndexLike | None = None,
    #         right_on: Scalar | IndexLike | None = None,
    #         left_index: bool = False,
    #         right_index: bool = False,
    #         suffixes: tuple[str, str]=("_x", "_y")
    #     ) -> DataFrame:
    #         return merge(self, right, how, on, left_on=left_on, right_on=right_on, left_index=left_index, right_index=right_index, suffixes=suffixes)

    #     ###########################################################################
    #     # Statistics
    #     ###########################################################################
    #     @overload
    #     def mean(self, axis: Axis) -> Series: ...  # no cov
    #     @overload
    #     def mean(self, axis: None) -> Scalar: ...  # no cov
    #     def mean(self, axis: AxisOrNone = 0) -> Series | Scalar:
    #         """
    #         Computes the mean of the Series.

    #         Args:
    #             axis: AxisOrNone to aggregate along:
    #                 - 0: Aggregate each column (default)
    #                 - 1: Aggregate each row
    #                 - None: Aggregates along both axes returning a scalar

    #         Returns:
    #             Series | float: Axis mean
    #         """
    #         return self._agg_with_none(statistics.mean, axis=axis)

    #     @overload
    #     def median(self, axis: Axis) -> Series: ...  # no cov
    #     @overload
    #     def median(self, axis: None) -> Scalar: ...  # no cov
    #     def median(self, axis: AxisOrNone = 0) -> Series | Scalar:
    #         """
    #         Return the median (middle value) of numeric data, using the common “mean of middle two” method.
    #         If data is empty, StatisticsError is raised. data can be a sequence or iterable.

    #         Args:
    #             axis: AxisOrNone to aggregate along:
    #                 - 0: Aggregate each column (default)
    #                 - 1: Aggregate each row
    #                 - None: Aggregates along both axes returning a scalar

    #         Returns:
    #             Series | float | int: Axis median
    #         """
    #         return self._agg_with_none(statistics.median, axis=axis)

    #     def mode(self, axis: Axis = 0) -> Series:
    #         """
    #         Return the single most common data point from discrete or nominal data. The mode (when it exists)
    #         is the most typical value and serves as a measure of central location.

    #         Args:
    #             axis: Axis to aggregate along:
    #                 - 0: Aggregate each column (default)
    #                 - 1: Aggregate each row

    #         Returns:
    #             Series: Axis mode
    #         """
    #         # @TOOO: Improve this. Might have to implement NaNs
    #         return self.agg(statistics.mode, axis=axis)

    #     def quantiles(self, *, n=4, method: Literal["exclusive", "inclusive"] = "exclusive", axis: Axis = 0) -> Series:
    #         """
    #         Divide data into n continuous intervals with equal probability. Returns a list of `n - 1`
    #         cut points separating the intervals.

    #         Args:
    #             axis: AxisOrNone to aggregate along:
    #                 - 0: Aggregate each column (default)
    #                 - 1: Aggregate each row

    #         Returns:
    #             Series: Series of lists containing quantiles
    #         """
    #         return self.agg(lambda values: statistics.quantiles(values, n=n, method=method), axis=axis)

    #     def std(self, xbar=None, axis: Axis = 0) -> Series | Scalar:
    #         """
    #         Return the sample standard deviation (the square root of the sample variance).
    #         See variance() for arguments and other details.

    #         Args:
    #             axis: AxisOrNone to aggregate along:
    #                 - 0: Aggregate each column (default)
    #                 - 1: Aggregate each row

    #         Returns:
    #             Series: Standard deviations along axis
    #         """
    #         return self.agg(lambda values: statistics.stdev(values, xbar=xbar), axis=axis)

    #     def var(self, xbar=None, axis: Axis = 0) -> Series | Scalar:
    #         """
    #         Return the sample variance of data, an iterable of at least two real-valued numbers.
    #         Variance, or second moment about the mean, is a measure of the variability
    #         (spread or dispersion) of data. A large variance indicates that the data is spread out;
    #         a small variance indicates it is clustered closely around the mean.

    #         Args:
    #             axis: Axis to aggregate along:
    #                 - 0: Aggregate each column (default)
    #                 - 1: Aggregate each row

    #         Returns:
    #             Series: Variances along axis
    #         """
    #         return self._agg_with_none(lambda values: statistics.variance(values, xbar=xbar), axis=axis)

    #     ###########################################################################
    #     # Exports
    #     ###########################################################################
    # def to_list(self) -> list[list[Any]]:
    #     """
    #     Converts the DataFrame to a list.

    #     Returns:
    #         list[list[Any]]: A list of the Series values.
    #     """
    #     return list(self.apply(lambda s: s.values, axis=1).data.values())

    #     @overload
    #     def to_dict(self) -> dict[Scalar, dict[Scalar, Any]]: ...  # no cov
    #     @overload
    #     def to_dict(self, orient: Literal["dict"]) -> dict[Scalar, dict[Scalar, Any]]: ...  # no cov
    #     @overload
    #     def to_dict(self, orient: Literal["list"]) -> dict[Scalar, list[Any]]: ...  # no cov
    #     @overload
    #     def to_dict(self, orient: Literal["records"]) -> list[dict[Scalar, Any]]: ...  # no cov
    #     def to_dict(self, orient: Literal["dict", "list", "records"] = "dict"):
    #         """
    #         Converts the DataFrame to a dictionary.

    #         Args:
    #             orient str {`dict`, `list`, `records`}: Determines the type of the values of the
    #                 dictionary.

    #         Returns:
    #             dict[Scalar, Any]: A dictionary representation of the Series.
    #         """
    #         match orient:
    #             case "dict":
    #                 return self.apply(lambda s: s.to_dict()).data
    #             case "list":
    #                 return self.apply(lambda s: s.to_list()).data
    #             case "records":
    #                 return list(self.apply(lambda s: s.to_dict(), axis=1).values)
    #             case _:
    #                 msg = f"orient '{orient}' not understood"
    #                 raise ValueError(msg)

    ###########################################################################
    # Comparisons
    ###########################################################################
    def _op(
        self, op: str, other: DataFrame | Series | Mapping | ArrayLike | Scalar
    ) -> DataFrame:  # @TODO: Implement axis
        match other:
            case DataFrame():
                return self._op_dataframe(op, other)
            case Series():
                return self._op_series(op, other)
            case ArrayLike() | Mapping() as c if len(c) == 0 and len(self) == 0:
                return DataFrame()
            case ArrayLike() | Mapping() as c if len(c) != len(self.columns):
                msg = f"Unable to coerce to Series, length must be {len(self.columns)}: given {len(other)}"
                raise ValueError(msg)
            case Mapping():
                return self._op_series(op, Series(other))
            case ArrayLike() as c if not isinstance(c, str):
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
        if set(self.columns) != set(other.columns):
            msg = "Can only compare identically-labeled (both index and columns) DataFrame objects"
            raise ValueError(msg)
        return DataFrame(
            (getattr(s, op)(other.loc[idx]) for idx, s in self.iterrows()), index=self.index, columns=self.columns
        )

    def _op_scalar(self, op: str, other: ArrayLike | Scalar) -> DataFrame:
        return DataFrame((getattr(s, op)(other) for _, s in self.iterrows()))

    def __lt__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:  # type: ignore
        """
        Element-wise less than comparison.

        Compares each value in the DataFrame with the corresponding value in `other`.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame Series,
                ArrayLike, or Scalar to compare with.

        Returns:
            DataFrame: A DataFrame of boolean values indicating the result of the comparison.
        """
        return self._op("__lt__", other)

    def __le__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:  # type: ignore
        """
        Element-wise less than or equal to comparison.

        Compares each value in the DataFrame with the corresponding value in `other`.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame Series,
                ArrayLike, or Scalar to compare with.

        Returns:
            DataFrame: A DataFrame of boolean values indicating the result of the comparison.
        """
        return self._op("__le__", other)

    def __eq__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:  # type: ignore
        """
        Element-wise equality comparison.

        Compares each value in the DataFrame with the corresponding value in `other`.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame Series,
                ArrayLike, or Scalar to compare with.

        Returns:
            DataFrame: A DataFrame of boolean values indicating the result of the comparison.
        """
        return self._op("__eq__", other)

    def __ne__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:  # type: ignore
        """
        Element-wise inequality comparison.

        Compares each value in the DataFrame with the corresponding value in `other`.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame Series,
                ArrayLike, or Scalar to compare with.

        Returns:
            DataFrame: A DataFrame of boolean values indicating the result of the comparison.
        """
        return self._op("__ne__", other)

    def __gt__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:  # type: ignore
        """
        Element-wise greater than comparison.

        Compares each value in the DataFrame with the corresponding value in `other`.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame Series,
                ArrayLike, or Scalar to compare with.

        Returns:
            DataFrame: A DataFrame of boolean values indicating the result of the comparison.
        """
        return self._op("__gt__", other)

    def __ge__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:  # type: ignore
        """
        Element-wise greater than or equal to comparison.

        Compares each value in the DataFrame with the corresponding value in `other`.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame Series,
                ArrayLike, or Scalar to compare with.

        Returns:
            DataFrame: A DataFrame of boolean values indicating the result of the comparison.
        """
        return self._op("__ge__", other)

    ###########################################################################
    # Operators
    ###########################################################################
    def __add__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        """
        Element-wise addition.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame, Series, ArrayLike, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self._op("__add__", other)

    def __sub__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        """
        Element-wise subtraction.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame, Series, ArrayLike, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self._op("__sub__", other)

    def __mul__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        """
        Element-wise multiplication.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame, Series, ArrayLike, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self._op("__mul__", other)

    def __matmul__(self, other: DataFrame | Series | ArrayLike) -> DataFrame | Series:
        """
        Performs dot product with another Series, ArrayLike or Scalar.

        If other is a Series or a ArrayLike, performs the dot product between the two.
        If other is a Scalar, multiplies all elements of the Series by the scalar and returns the sum.

        Args:
            other (Series | ArrayLike | Scalar)

        Returns:
            Scalar: The dot product of the Series.
        """
        return self.dot(other)

    def __truediv__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        """
        Element-wise division.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame, Series, ArrayLike, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self._op("__truediv__", other)

    def __floordiv__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        """
        Element-wise floor division.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame, Series, ArrayLike, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self._op("__floordiv__", other)

    def __mod__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        """
        Element-wise modulo.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame, Series, ArrayLike, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self._op("__mod__", other)

    def __divmod__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        """
        Element-wise divmod.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame, Series, ArrayLike, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self._op("__divmod__", other)

    def __pow__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        """
        Element-wise exponentiation.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame, Series, ArrayLike, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self._op("__pow__", other)

    def __lshift__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        """
        Element-wise left bit shift.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame, Series, ArrayLike, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self._op("__lshift__", other)

    def __rshift__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        """
        Element-wise right bit shift.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame, Series, ArrayLike, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self._op("__rshift__", other)

    def __and__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        """
        Element-wise AND.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame, Series, ArrayLike, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self._op("__and__", other)

    def __xor__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        """
        Element-wise XOR.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame, Series, ArrayLike, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self._op("__xor__", other)

    def __or__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        """
        Element-wise OR.

        Args:
            other (DataFrame | Series | ArrayLike | Scalar): The other DataFrame, Series, ArrayLike, or Scalar to operate with.

        Returns:
            DataFrame: A DataFrame with the results of the operation.
        """
        return self._op("__or__", other)

    ###########################################################################
    # Right-hand Side Operators
    ###########################################################################
    def __radd__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        return self._op("__radd__", other)

    def __rsub__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        return self._op("__rsub__", other)

    def __rmul__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        return self._op("__rmul__", other)

    def __rmatmul__(self, other: ArrayLike) -> Series:
        if len(self.index) != len(other):
            msg = f"shapes {Series(other).shape} and {self.shape} not aligned"
            raise ValueError(msg)
        other = Series(other, index=self.index)
        data = [other.dot(s_a) for s_a in self.T.values]
        return Series(data, index=self.columns, name=other.name)

    def __rtruediv__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        return self._op("__rtruediv__", other)

    def __rfloordiv__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        return self._op("__rfloordiv__", other)

    def __rmod__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        return self._op("__rmod__", other)

    def __rdivmod__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        return self._op("__rdivmod__", other)

    def __rpow__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        return self._op("__rpow__", other)

    def __rlshift__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        return self._op("__rlshift__", other)

    def __rrshift__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        return self._op("__rrshift__", other)

    def __rand__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        return self._op("__rand__", other)

    def __rxor__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        return self._op("__rxor__", other)

    def __ror__(self, other: DataFrame | Series | ArrayLike | Scalar) -> DataFrame:
        return self._op("__ror__", other)

    ###########################################################################
    # In-place Operators
    ###########################################################################
    def _iop(self, op: str, other: DataFrame | Series | Mapping | ArrayLike | Scalar) -> Self:
        match other:
            case DataFrame():
                return self._iop_dataframe(op, other)
            case Series():
                return self._iop_series(op, other)
            case ArrayLike() | Mapping() as c if len(c) == 0 and len(self) == 0:
                return self
            case ArrayLike() | Mapping() as c if len(c) != len(self.columns):
                msg = f"Unable to coerce to Series, length must be {len(self.columns)}: given {len(other)}"
                raise ValueError(msg)
            case Mapping():
                return self._iop_series(op, Series(other))
            case ArrayLike() as c if not isinstance(c, str):
                return self._iop_series(op, Series(other, index=self.columns))
            # No 2d collection comparison. Will consider 2d inputs as a series of collections
            case _:  # Everithing else is a scalar then
                return self._iop_scalar(op, other)

    def _iop_series(self, op: str, other: Series) -> Self:
        if len(self.columns) != len(other):
            msg = "Operands are not aligned. Do `left, right = left.align(right, axis=1, copy=False)` before operating."
            raise ValueError(msg)
        for col, s in self.iterrows():
            getattr(s, op)(other[col])
        return self

    def _iop_dataframe(self, op: str, other: DataFrame) -> Self:
        if set(self.columns) != set(other.columns) or self.shape != other.shape:
            msg = "Can only compare identically-labeled (both index and columns) DataFrame objects"
            raise ValueError(msg)
        for col, s in self.iterrows():
            getattr(s, op)(other[col])
        return self

    def _iop_scalar(self, op: str, other: ArrayLike[Any] | Scalar) -> Self:
        for _, s in self.iterrows():
            getattr(s, op)(other)
        return self

    def __iadd__(self, other: DataFrame | Series | ArrayLike | Scalar) -> Self:
        return self._iop("__iadd__", other)

    def __isub__(self, other: DataFrame | Series | ArrayLike | Scalar) -> Self:
        return self._iop("__isub__", other)

    def __imul__(self, other: DataFrame | Series | ArrayLike | Scalar) -> Self:
        return self._iop("__imul__", other)

    # @TODO: How to overload correctly __matmul__ and __imatmul__?
    # @overload
    # def __imatmul__(self, other: DataFrame) -> DataFrame: ...  # no cov
    # @overload
    # def __imatmul__(self, other: Series | ArrayLike) -> Series: ...  # no cov
    def __imatmul__(self, other: DataFrame | Series | ArrayLike) -> DataFrame | Series:  # noqa: PYI034
        return self.dot(other)

    def __itruediv__(self, other: DataFrame | Series | ArrayLike | Scalar) -> Self:
        return self._iop("__itruediv__", other)

    def __ifloordiv__(self, other: DataFrame | Series | ArrayLike | Scalar) -> Self:
        return self._iop("__ifloordiv__", other)

    def __imod__(self, other: DataFrame | Series | ArrayLike | Scalar) -> Self:
        return self._iop("__imod__", other)

    def __ipow__(self, other: DataFrame | Series | ArrayLike | Scalar) -> Self:
        return self._iop("__ipow__", other)

    def __ilshift__(self, other: DataFrame | Series | ArrayLike | Scalar) -> Self:
        return self._iop("__ilshift__", other)

    def __irshift__(self, other: DataFrame | Series | ArrayLike | Scalar) -> Self:
        return self._iop("__irshift__", other)

    def __iand__(self, other: DataFrame | Series | ArrayLike | Scalar) -> Self:
        return self._iop("__iand__", other)

    def __ixor__(self, other: DataFrame | Series | ArrayLike | Scalar) -> Self:
        return self._iop("__ixor__", other)

    def __ior__(self, other: DataFrame | Series | ArrayLike | Scalar) -> Self:  # type: ignore
        return self._iop("__ior__", other)

    ###########################################################################
    # Unary Operators
    ###########################################################################
    def __neg__(self) -> DataFrame:
        return DataFrame([-s for s in self.values], index=self.index, columns=self.columns)  # type: ignore

    def __pos__(self) -> DataFrame:
        return DataFrame([+s for s in self.values], index=self.index, columns=self.columns)  # type: ignore

    def __abs__(self) -> DataFrame:
        return self.abs()

    def __invert__(self) -> DataFrame:
        return DataFrame([~s for s in self.values], index=self.index, columns=self.columns)  # type: ignore
