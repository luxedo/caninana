# SPDX-FileCopyrightText: 2025-present Luiz Eduardo Amaral <luizamaral306@gmail.com>
#
# SPDX-License-Identifier: MIT


import pandas as pd

import lontras as lt

from .assertions import assert_index_equal_pandas


class TestIndexInit:
    def test_init_empty(self):
        i = lt.Index([])
        pi = pd.Index([])
        assert_index_equal_pandas(i, pi)

    def test_init_integer(self):
        index_array = [0, 1, 1, 2, 3]
        i = lt.Index(index_array)
        pi = pd.Index(index_array)
        assert_index_equal_pandas(i, pi)
