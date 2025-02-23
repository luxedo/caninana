# SPDX-FileCopyrightText: 2025-present Luiz Eduardo Amaral <luizamaral306@gmail.com>
#
# SPDX-License-Identifier: MIT

import pytest

import lontras as lt
from lontras.lontras import _is_array_like, _is_boolean_mask


class TestTypeGuards:
    @pytest.mark.parametrize(
        "case",
        [
            [0, 1, 2],
            [True, False, False, int],
            lt.Array(["a", True]),
            lt.Series(["a", True]),
            (1, 2, 3),
            range(10),
        ],
    )
    def test_is_array_like(self, case):
        assert _is_array_like(case)

    @pytest.mark.parametrize(
        "case",
        [
            True,
            int,
            "hello",
            123,
        ],
    )
    def test_not_is_array_like(self, case):
        assert not _is_array_like(case)

    @pytest.mark.parametrize(
        "case",
        [
            [0, 1, 2],
            [True, False, False, int],
            lt.Array(["a", True]),
            lt.Series(["a", True]),
        ],
    )
    def test_not_is_boolean_mask(self, case):
        assert not _is_boolean_mask(case)

    @pytest.mark.parametrize(
        "case",
        [
            [True, False, False],
            lt.Array([False, True]),
            lt.Series([True, False]),
        ],
    )
    def test_is_boolean_mask(self, case):
        assert _is_boolean_mask(case)
