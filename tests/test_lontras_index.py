# SPDX-FileCopyrightText: 2025-present Luiz Eduardo Amaral <luizamaral306@gmail.com>
#
# SPDX-License-Identifier: MIT


import pandas as pd
import pytest

import lontras as lt

from .assertions import assert_exception, assert_index_equal_pandas

example_index = [0, 1, 2, 3]
example_label_index = ["a", "b", "c", "d"]


class TestIndexInit:
    def test_init_empty(self):
        i = lt.Index([])
        pi = pd.Index([])
        assert_index_equal_pandas(i, pi)

    def test_init_integer(self):
        i = lt.Index(example_index)
        pi = pd.Index(example_index)
        assert_index_equal_pandas(i, pi)

    def test_init_rename(self):
        name = "rename"
        new_name = "new_rename"
        i = lt.Index(example_index, name=name)
        ir = lt.Index(i, name=new_name)
        assert i.name == name
        assert ir.name == new_name

    def test__repr__(self):
        s = lt.Index([])
        assert str(s) == "Index([])"
        s = lt.Index(example_index)
        assert str(s) == f"Index({example_index!s})"
        example_name = "my_index"
        s = lt.Index([], name=example_name)
        assert str(s) == f'Index([], name="{example_name}")'
        example_name = "my_index"
        s = lt.Index(example_index, name=example_name)
        assert str(s) == f'Index({example_index!s}, name="{example_name}")'


class TestIndexAccessors:
    def test_setitem(self):
        i = lt.Index(example_label_index)
        pi = pd.Index(example_label_index)

        def pandas_error():
            pi[[0, 1]] = ["e", "f"]

        def lontras_error():
            i[[0, 1]] = ["e", "f"]

        assert_exception(pandas_error, lontras_error, TypeError)

    def test_delitem(self):
        i = lt.Index(example_label_index)
        pi = pd.Index(example_label_index)

        # Pandads does not throw the same error for __getitem__ and __delitem__.
        # We're chosing the __setitem__ message for __getitem__ too.
        def pandas_error():
            pi[[0, 1]] = ["e", "f"]

        def lontras_error():
            del i[[0, 1]]

        assert_exception(pandas_error, lontras_error, TypeError)


class TestIndexGetIlocs:
    def test_get_ilocs_series(self):
        i = lt.Index(example_label_index)
        indexes = ["b", "d"]
        positions = [example_label_index.index(i) for i in indexes]
        assert i.get_ilocs(lt.Series(indexes)) == positions

    def test_get_ilocs_array(self):
        i = lt.Index(example_label_index)
        indexes = ["b", "d"]
        positions = [example_label_index.index(i) for i in indexes]
        assert i.get_ilocs(lt.Array(indexes)) == positions

    def test_get_ilocs_list(self):
        i = lt.Index(example_label_index)
        indexes = ["b", "d"]
        positions = [example_label_index.index(i) for i in indexes]
        assert i.get_ilocs(indexes) == positions

    def test_get_ilocs_slice(self):
        i = lt.Index(example_label_index)
        indexes = slice(1, 2)
        positions = [example_label_index.index(i) for i in example_label_index[indexes]]
        assert i.get_ilocs(indexes) == positions

    def test_get_ilocs_scalar(self):
        i = lt.Index(example_label_index)
        index = "c"
        position = example_label_index.index(index)
        assert i.get_ilocs(index) == position

    def test_get_ilocs_scalar_duplicated(self):
        dup_index = [*example_label_index, "c"]
        i = lt.Index(dup_index)
        index = "c"
        positions = [i for i, idx in enumerate(dup_index) if idx == index]
        assert i.get_ilocs(index) == positions

    def test_get_ilocs_error(self):
        i = lt.Index(example_label_index)
        with pytest.raises(KeyError, match="Cannot index with:"):
            i.get_ilocs(int)


class TestIndexComparison:
    @pytest.mark.parametrize(
        ("ia", "ib"),
        [
            (lt.Index([]), lt.Index([])),
            (lt.Index(example_index), lt.Index(example_index)),
            (lt.Index(example_index), example_index),
        ],
    )
    def test_eq(self, ia, ib):
        assert (ia == ib).all()

    @pytest.mark.parametrize(
        ("ia", "ib"),
        [
            (lt.Index(example_index), lt.Index(reversed(example_index))),
            (lt.Index(reversed(example_index)), example_index),
        ],
    )
    def test_ne(self, ia, ib):
        assert (ia != ib).all()
