# SPDX-FileCopyrightText: 2025-present Luiz Eduardo Amaral <luizamaral306@gmail.com>
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

import lontras as lt

if TYPE_CHECKING:
    from collections.abc import Callable


def assert_exception(fn_src: Callable, fn_dst: Callable, exc_type: type[Exception], match: str = ""):
    try:
        fn_src()
    except exc_type as e:
        with pytest.raises(exc_type, match=match if match else re.escape(str(e))):
            fn_dst()
    else:
        pytest.fail(f"fn_src did not raise an exception of type {exc_type.__name__}")  # no cover


def assert_array_equal_numpy(a: lt.Array, na: np.ndarray):
    assert isinstance(a, lt.Array)
    assert isinstance(na, np.ndarray)
    assert (a == na).all()


# def assert_dataframe_equal_pandas(df: lt.DataFrame, pdf: pd.DataFrame):
#     assert isinstance(s, lt.DataFrame)
#     assert isinstance(ps, pd.DataFrame)
#     assert df.shape == pdf.shape
#     assert all(df.columns == pdf.columns) is True  # type: ignore
#     assert all(df.index == pdf.index) is True  # type: ignore
#     for col in df.columns:
#         assert_series_equal_pandas(df[col], pdf[col])


def assert_series_equal_pandas(s: lt.Series, ps: pd.Series):
    assert isinstance(s, lt.Series)
    assert isinstance(ps, pd.Series)
    assert len(s) == len(ps)
    assert s.name == ps.name
    assert_index_equal_pandas(s.index, ps.index)
    assert (s == ps).all() is True
    assert (ps.to_dict() == s.to_dict()) is True


def assert_index_equal_pandas(i: lt.Index, pi: pd.Index):
    assert len(i) == len(pi)
    assert i.name == pi.name
    assert (i.values == pi.values.tolist()) is True


# def assert_scalar_equal(v0, v1):
#     assert _is_scalar(v0)
#     assert _is_scalar(v1)
#     assert v0 == v1
