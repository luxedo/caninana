# SPDX-FileCopyrightText: 2025-present Luiz Eduardo Amaral <luizamaral306@gmail.com>
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    import lontras as lt


def assert_exception(fn_src: Callable, fn_dst: Callable, exc_type: type[Exception], match: str = ""):
    try:
        fn_src()
    except exc_type as e:
        with pytest.raises(exc_type, match=match if match else re.escape(str(e))):
            fn_dst()
    else:
        pytest.fail(f"fn_src did not raise an exception of type {exc_type.__name__}")  # no cover


def assert_dataframe_equal_pandas(df: lt.DataFrame, pdf: pd.DataFrame):
    assert df.shape == pdf.shape
    assert all(df.columns == pdf.columns) is True  # type: ignore
    assert all(df.index == pdf.index) is True  # type: ignore
    for col in df.columns:
        assert_series_equal_pandas(df[col], pdf[col])


def assert_series_equal_pandas(s: lt.Series, ps: pd.Series):
    assert len(s) == len(ps)
    assert (list(s.index) == list(ps.index)) is True
    assert s.name == ps.name
    assert (s == ps).all() is True
    assert (ps.to_dict() == s.to_dict()) is True


def assert_scalar_equal(v0, v1):
    assert v0 == v1
