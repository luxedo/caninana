# SPDX-FileCopyrightText: 2025-present Luiz Eduardo Amaral <luizamaral306@gmail.com>
#
# SPDX-License-Identifier: MIT

import statistics

import pandas as pd
import pytest

import lontras as lt

from .assertions import assert_dataframe_equal_pandas, assert_exception, assert_scalar_equal, assert_series_equal_pandas

example_list_dict = [{"a": 0, "b": 1}, {"a": 3, "b": 4}, {"a": 6, "b": 7}]
example_list_dict_dataframe_str = """   a  b
0  0  1
1  3  4
2  6  7
"""
example_list_series = [lt.Series({"a": 0, "b": 1}), lt.Series({"a": 3, "b": 4}), lt.Series({"a": 6, "b": 7})]
example_index = ["d", "e", "f"]
example_columns = [3, 4]
example_array = [[0, 1], [3, 4], [6, 7]]
example_collection_scalars = [0, 1, 2]
# example_series = lt.Series([10, 11], index=example_index)
example_dict_list = {"col_a": [1, 2, 3], "col_b": [4, 5, 6]}
example_dict_series = {
    "col_a": lt.Series([1, 2, 3], index=example_index),
    "col_b": lt.Series([4, 5, 6], index=example_index),
}
example_dict_series_pd = {
    "col_a": pd.Series([1, 2, 3], index=example_index),
    "col_b": pd.Series([4, 5, 6], index=example_index),
}
example_scalar = 3
example_cmp_a = [0, 1, 3]
example_cmp_b = [1, 2, 3]
example_unary = [-3, -1, 0, 1, 2]


class TestDataFrameInit:
    def test_init_none(self):
        df = lt.DataFrame()
        pdf = pd.DataFrame()
        assert_dataframe_equal_pandas(df, pdf)

    def test_init_empty_mapping(self):
        df = lt.DataFrame({})
        pdf = pd.DataFrame({})
        assert_dataframe_equal_pandas(df, pdf)

    def test_init_empty_collection(self):
        df = lt.DataFrame([])
        pdf = pd.DataFrame([])
        assert_dataframe_equal_pandas(df, pdf)
        df = lt.DataFrame(())
        pdf = pd.DataFrame(())
        assert_dataframe_equal_pandas(df, pdf)
        df = lt.DataFrame(set())
        pdf = pd.DataFrame(set())
        assert_dataframe_equal_pandas(df, pdf)

    def test_init_empty_2d_collection(self):
        df = lt.DataFrame([[]])
        # pdf = pd.DataFrame([[]])
        assert_dataframe_equal_pandas(df, pd.DataFrame())
        # Pandas does create a DataFrame with zero columns and one row
        # We're keeping it empty

    def test_init_empty_mapping_collection(self):
        df = lt.DataFrame({"a": lt.Series()})
        # pdf = pd.DataFrame([[]])
        assert_dataframe_equal_pandas(df, pd.DataFrame())
        # Pandas does create a DataFrame with zero columns and one row

    def test_init_from_mapping_of_collections(self):
        df = lt.DataFrame(example_dict_list)
        pdf = pd.DataFrame(example_dict_list)
        assert_dataframe_equal_pandas(df, pdf)

    def test_init_from_mapping_of_series(self):
        df = lt.DataFrame(example_dict_series)
        pdf = pd.DataFrame(example_dict_series_pd)
        assert_dataframe_equal_pandas(df, pdf)

    def test_init_collection_of_scalars(self):
        df = lt.DataFrame(example_collection_scalars)
        pdf = pd.DataFrame(example_collection_scalars)
        assert_dataframe_equal_pandas(df, pdf)
        df = lt.DataFrame(tuple(example_collection_scalars))
        pdf = pd.DataFrame(tuple(example_collection_scalars))
        assert_dataframe_equal_pandas(df, pdf)

    def test_init_from_collection_of_series(self):
        df = lt.DataFrame(example_list_series)
        pdf = pd.DataFrame(example_list_series)
        assert_dataframe_equal_pandas(df, pdf)

    def test_init_from_collection_of_mappings(self):
        df = lt.DataFrame(example_list_dict)
        pdf = pd.DataFrame(example_list_dict)
        assert_dataframe_equal_pandas(df, pdf)

    def test_init_from_collection_of_mappings_with_index(self):
        df = lt.DataFrame(example_list_dict, index=example_index)
        pdf = pd.DataFrame(example_list_dict, index=example_index)
        assert_dataframe_equal_pandas(df, pdf)

    def test_init_generator(self):
        df = lt.DataFrame(2**i for i in range(5))
        pdf = pd.DataFrame(2**i for i in range(5))
        assert_dataframe_equal_pandas(df, pdf)

    # This filters the data and keeps only the new columns with NaNs.
    # def test_init_from_collection_of_mappings_with_columns(self):
    #     df = lt.DataFrame(example_list_dict, columns=example_columns)
    #     pdf = pd.DataFrame(example_list_dict, columns=example_columns)
    #     assert_dataframe_equal_pandas(df, pdf)

    def test_init_from_2d_collection(self):
        df = lt.DataFrame(example_array)
        pdf = pd.DataFrame(example_array)
        assert_dataframe_equal_pandas(df, pdf)

    def test_init_from_2d_collection_with_index(self):
        df = lt.DataFrame(example_array, index=example_index)
        pdf = pd.DataFrame(example_array, index=example_index)
        assert_dataframe_equal_pandas(df, pdf)

    def test_init_from_2d_collection_with_columns(self):
        df = lt.DataFrame(example_array, columns=example_columns)
        pdf = pd.DataFrame(example_array, columns=example_columns)
        assert_dataframe_equal_pandas(df, pdf)

    # Pandas supports misshaped inputs
    # def test_init_constructor_error_misshaped_inputs(self):
    #     assert_exception(lambda: pd.DataFrame([[0, 1, 2], [0, 1]]),
    #                      lambda: lt.DataFrame([[0, 1, 2], [0, 1]]), ValueError)

    def test_init_constructor_error_scalar(self):
        assert_exception(lambda: pd.DataFrame("hello pandas"), lambda: lt.DataFrame("hello lontras"), ValueError)

    def test_init_constructor_error_empty_data_with_columns(self):
        match = "Cannot create an empty"
        with pytest.raises(ValueError, match=match):
            lt.DataFrame(index=example_index)
        with pytest.raises(ValueError, match=match):
            lt.DataFrame(columns=example_columns)

    def test_init_constructor_error_mapping_of_series_misaligned_indexes(self):
        assert_exception(
            lambda: pd.DataFrame([0, 1], index=[0]),
            lambda: lt.DataFrame([0, 1], index=[0]),
            ValueError,
        )
        # Using this initializer we've got no exceptions in pandas
        # pd.DataFrame({0: pd.Series([0, 1])}, index=[0]),
        # lt.DataFrame({0: lt.Series([0, 1])}, index=[0]),

    def test_init_constructor_error_mapping_of_series_misaligned_columns(self):
        assert_exception(
            lambda: pd.DataFrame([0, 1], columns=[0, 1]), lambda: lt.DataFrame([0, 1], columns=[0, 1]), ValueError
        )

    def test_init_constructor_error_2d_collection_misaligned_indexes(self):
        match = "Shape of passed values is"
        with pytest.raises(ValueError, match=match):
            lt.DataFrame([[0, 1], [2, 3]], index=[0])
        # Pandas raises a different exeption in this case. We'll keep the previous message
        # ValueError: Length of values (2) does not match length of index (1)
        # assert_exception(
        #     lambda: pd.DataFrame([[0, 1], [2, 3]], index=[0]),
        #     lambda: lt.DataFrame([[0, 1], [2, 3]], index=[0]),
        #     ValueError,
        # )

    def test_init_constructor_error_2d_collection_misaligned_columns(self):
        match = "Shape of passed values is"
        with pytest.raises(ValueError, match=match):
            lt.DataFrame([[0, 1], [2, 3]], columns=[0])
        # assert_exception(
        #     lambda: pd.DataFrame([[0, 1], [2, 3]], columns=[0]),
        #     lambda: lt.DataFrame([[0, 1], [2, 3]], columns=[0]),
        #     ValueError,
        # )
        # Pandas raises a different exeption in this case. We'll keep the previous message
        # AssertionError: 1 columns passed, passed data had 2 columns
        # '1 columns passed, passed data had 2 columns'

    def test_init_constructor_error_different_indexes_series(self):
        match = "Misaligned columns"
        with pytest.raises(ValueError, match=match):
            lt.DataFrame([lt.Series({0: 1, 1: 2}), lt.Series({3: 4, 5: 6})])

    def test_index_column_validator_index(self):
        df = lt.DataFrame(example_list_dict)
        df["a"] = df["a"].reindex([9, 8, 7])
        match = "Somehow the inner indexes"
        with pytest.raises(ValueError, match=match):
            df._validate_index_and_columns()  # noqa: SLF001

    def test_index_column_validator_columns(self):
        df = lt.DataFrame(example_list_dict)
        df["a"] = df["a"].rename("bob")
        match = "Somehow the inner columns"
        with pytest.raises(ValueError, match=match):
            df._validate_index_and_columns()  # noqa: SLF001

    def test__repr__(self):
        df = lt.DataFrame()
        assert str(df) == "Empty DataFrame"
        df = lt.DataFrame([])
        assert str(df) == "Empty DataFrame"
        df = lt.DataFrame([[]])
        assert str(df) == "Empty DataFrame"
        df = lt.DataFrame(example_list_dict)
        pdf = pd.DataFrame(example_list_dict)
        assert str(df) == str(pdf)
        df = lt.DataFrame(example_array)
        pdf = pd.DataFrame(example_array)
        assert str(df) == str(pdf)
        df = lt.DataFrame(example_dict_list)
        pdf = pd.DataFrame(example_dict_list)
        assert str(df) == str(pdf)
        df = lt.DataFrame(example_collection_scalars)
        pdf = pd.DataFrame(example_collection_scalars)
        assert str(df) == str(pdf)


#     def test_shallow_copy(self):
#         df = lt.DataFrame([[123]])
#         t = df.copy(deep=False)
#         df.iloc[0][0] = [456]
#         assert (t == s).all()

#     def test_deepcopy(self):
#         df = lt.DataFrame([[123]])
#         t = df.copy()
#         df.iloc[0][0] = [456]
#         assert (t != s).all()

#     def test_index_getter(self):
#         df = lt.DataFrame(example_list_dict)
#         assert df.index == list(example_index)

#     def test_index_setter(self):
#         df = lt.DataFrame(example_list_dict)
#         df.index = list(reversed(example_index))
#         assert df.index == list(reversed(example_index))

#     def test_index_setter_error(self):
#         df = lt.DataFrame(example_list_dict)
#         with pytest.raises(ValueError, match="Length mismatch"):
#             df.index = [*list(example_index), "more_indexes"]

# def test_T(self):
#     df = lt.DataFrame(example_list_dict)
#     pdf = pd.DataFrame(example_list_dict)
#     assert (df.T == pdf.T).all(axis=None)
#     assert (pdf.T == df.T).all(axis=None)


# class TestDataFrameAccessors:
#     def test_getitem_0(self):
#         df = lt.DataFrame(example_list_dict)
#         pdf = pd.DataFrame(example_list_dict)
#         key = "a"
#         assert s[key] == ps[key]
#         key = "b"
#         assert s[key] == ps[key]
#         key = "c"
#         assert s[key] == ps[key]

#     def test_getitem_1(self):
#         df = lt.DataFrame(example_list_dict)
#         pdf = pd.DataFrame(example_list_dict)
#         indexedf = ["a", "c"]
#         assert (s[indexes] == ps[indexes]).all()

#     def test_getitem_2(self):
#         df = lt.DataFrame(example_list_dict)
#         pdf = pd.DataFrame(example_list_dict)
#         indexedf = slice(0, 2)
#         assert (s[indexes] == ps[indexes]).all()

#     def test_getitem_3(self):
#         df = lt.DataFrame(example_list_dict)
#         pdf = pd.DataFrame(example_list_dict)
#         val = 2
#         mask_df = s > val
#         mask_pdf = ps > val
#         assert (s[mask_s] == ps[mask_ps]).all()

#     def test_getitem_4(self):
#         df = lt.DataFrame(example_list_dict)
#         pdf = pd.DataFrame(example_list_dict)
#         indexedf = lt.DataFrame(example_index[:2])
#         assert (s[indexes] == ps[indexes]).all()

#     def test_loc_getitem_scalar(self):
#         df = lt.DataFrame(example_list_dict)
#         pdf = pd.DataFrame(example_list_dict)
#         key = "a"
#         assert df.loc[key] == pdf.loc[key]
#         key = "b"
#         assert df.loc[key] == pdf.loc[key]
#         key = "c"
#         assert df.loc[key] == pdf.loc[key]

#     def test_loc_setitem_scalar(self):
#         df = lt.DataFrame(example_list_dict)
#         pdf = pd.DataFrame(example_list_dict)
#         key = "a"
#         value = 4
#         df.loc[key] = value
#         pdf.loc[key] = value
#         assert df.loc[key] == ps[key]

#     def test_loc_getitem_list(self):
#         df = lt.DataFrame(example_list_dict)
#         pdf = pd.DataFrame(example_list_dict)
#         keydf = ["a", "b"]
#         assert (df.loc[keys] == pdf.loc[keys]).all()

#     def test_loc_setitem_list(self):
#         df = lt.DataFrame(example_list_dict)
#         pdf = pd.DataFrame(example_list_dict)
#         keydf = ["a", "b"]
#         value = 4
#         df.loc[keys] = value
#         pdf.loc[keys] = value
#         assert (df.loc[keys] == pdf.loc[keys]).all()

#     def test_loc_get_not_hashable_key(self):
#         df = lt.DataFrame(example_list_dict)
#         with pytest.raises(TypeError, match="Cannot index"):
#             df.loc[{1, 2, 3}]

#     def test_loc_set_not_hashable_key(self):
#         df = lt.DataFrame(example_list_dict)
#         with pytest.raises(TypeError, match="Cannot index"):
#             df.loc[{1, 2, 3}] = "no!"

#     def test_iloc_getitem_scalar(self):
#         df = lt.DataFrame(example_list_dict)
#         pdf = pd.DataFrame(example_list_dict)
#         index = 0
#         assert df.iloc[index] == pdf.iloc[index]
#         index = 1
#         assert df.iloc[index] == pdf.iloc[index]
#         index = 2
#         assert df.iloc[index] == pdf.iloc[index]

#     def test_iloc_setitem_scalar(self):
#         df = lt.DataFrame(example_list_dict)
#         pdf = pd.DataFrame(example_list_dict)
#         index = 0
#         value = 4
#         df.iloc[index] = value
#         pdf.iloc[index] = value
#         assert df.iloc[index] == pdf.iloc[index]

#     def test_iloc_getitem_slice(self):
#         df = lt.DataFrame(example_list_dict)
#         pdf = pd.DataFrame(example_list_dict)
#         indexedf = slice(0, 2, 1)
#         assert (df.iloc[indexes] == pdf.iloc[indexes]).all()

#     def test_iloc_setitem_slice(self):
#         df = lt.DataFrame(example_list_dict)
#         pdf = pd.DataFrame(example_list_dict)
#         indexedf = slice(0, 2, 1)
#         value = 4
#         df.iloc[indexes] = value
#         pdf.iloc[indexes] = value
#         assert (df.iloc[indexes] == pdf.iloc[indexes]).all()

#     def test_iloc_getitem_list(self):
#         df = lt.DataFrame(example_list_dict)
#         pdf = pd.DataFrame(example_list_dict)
#         indexedf = [0, 1]
#         assert (df.iloc[indexes] == pdf.iloc[indexes]).all()

#     def test_iloc_setitem_list(self):
#         df = lt.DataFrame(example_list_dict)
#         pdf = pd.DataFrame(example_list_dict)
#         indexedf = [0, 1]
#         value = 4
#         df.iloc[indexes] = value
#         pdf.iloc[indexes] = value
#         assert (df.iloc[indexes] == pdf.iloc[indexes]).all()

#     def test_iloc_get_error(self):
#         df = lt.DataFrame(example_list_dict)
#         with pytest.raises(TypeError, match="Cannot index"):
#             df.iloc[{1, 2, 3}]

#     def test_loc_set_error(self):
#         df = lt.DataFrame(example_list_dict)
#         with pytest.raises(TypeError, match="Cannot index"):
#             df.iloc[{1, 2, 3}] = "no!"

#     def test_head(self):
#         df = lt.DataFrame(example_list_dict)
#         pdf = pd.DataFrame(example_list_dict)
#         n = 2
#         assert len(df.head(n)) == n
#         assert (df.head(n) == example_values[:n]).all()
#         assert (df.head(n) == pdf.head(n)).all()

#     def test_tail(self):
#         df = lt.DataFrame(example_list_dict)
#         pdf = pd.DataFrame(example_list_dict)
#         n = 2
#         assert len(df.tail(n)) == n
#         assert (df.tail(n) == example_values[-n:]).all()
#         assert (df.tail(n) == pdf.tail(n)).all()

#     def test_find(self):
#         df = lt.DataFrame(example_list_dict)
#         key = "a"
#         assert df.find(example_list_dict[key]) == key

#     def test_find_not_found(self):
#         df = lt.DataFrame(example_list_dict)
#         assert df.find("value not found") is None

#     def test_ifind(self):
#         df = lt.DataFrame(example_list_dict)
#         index = 0
#         key = list(example_index)[index]
#         assert df.ifind(example_list_dict[key]) == index

#     def test_ifind_not_found(self):
#         df = lt.DataFrame(example_list_dict)
#         assert df.ifind("value not found") is None


class TestDataFrameMapReduce:
    #     def test_map(self):
    #         df = lt.DataFrame(example_list_dict)
    #         pdf = pd.DataFrame(example_list_dict)
    #         assert (df.map(lambda x: x**2) == pdf.map(lambda x: x**2)).all()

    #     def test_reduce(self):
    #         df = lt.DataFrame(example_list_dict)
    #         assert df.reduce(lambda acc, cur: acc + cur[0], "") == "".join(example_index)
    #         df = lt.DataFrame()
    #         assert df.reduce(lambda *_: 0, 0) == 0

    @pytest.mark.parametrize(
        "func",
        [
            "max",
            "min",
            "sum",
            "all",
            "any",
            "idxmax",
            "idxmin",
        ],
    )
    def test_aggregations(self, func):
        df = lt.DataFrame(example_list_dict)
        pdf = pd.DataFrame(example_list_dict)
        assert_series_equal_pandas(getattr(df, func)(axis=0), getattr(pdf, func)(axis=0))
        assert_series_equal_pandas(getattr(df, func)(axis=1), getattr(pdf, func)(axis=1))

    @pytest.mark.parametrize(
        "func",
        [
            "max",
            "min",
            "sum",
            "all",
            "any",
        ],
    )
    def test_aggregations_with_none(self, func):
        df = lt.DataFrame(example_list_dict)
        pdf = pd.DataFrame(example_list_dict)
        if func == "sum":
            # https://pandadf.pydata.org/docs/reference/api/pandadf.DataFrame.sum.html
            # Pandas sum behavior is odd :(
            assert getattr(df, func)(axis=None) == pdf.sum().sum()
        else:
            assert_scalar_equal(getattr(df, func)(axis=None), getattr(pdf, func)(axis=None))

    def test_agg_wrong_axis(self):
        match = "No axis named"
        with pytest.raises(ValueError, match=match):
            lt.DataFrame(example_list_dict).agg(lambda x: x, axis=-1)

    def test_apply_wrong_axis(self):
        match = "No axis named"
        with pytest.raises(ValueError, match=match):
            lt.DataFrame(example_list_dict).apply(lambda x: x, axis=-1)

    # def test_any(self):
    #     df = lt.DataFrame([0, 1, 2])
    #     pdf = pd.DataFrame([0, 1, 2])
    #     assert df.any() == pdf.any()
    #     df = lt.DataFrame([0])
    #     pdf = pd.DataFrame([0])
    #     assert df.any() == pdf.any()

    def test_astype(self):
        df = lt.DataFrame(example_list_dict)
        pdf = pd.DataFrame(example_list_dict)
        assert_dataframe_equal_pandas(df.astype(str), pdf.astype(str))

    def test_abs(self):
        df = lt.DataFrame(example_list_dict)
        pdf = pd.DataFrame(example_list_dict)
        assert_dataframe_equal_pandas(df.abs(), pdf.abs())


#     @pytest.mark.parametrize(
#         "func",
#         [
#             "argmax",
#             "argmin",
#             "idxmax",
#             "idxmin",
#         ],
#     )
#     def test_arg_idx_errors(self, func):
#         df = lt.DataFrame()
#         with pytest.raises(ValueError, match="empty sequence"):
#             getattr(s, func)()


class TestDataFrameStatistics:
    @pytest.mark.parametrize(
        "func",
        [
            "mean",
            "median",
            # "mode",
            "std",
            "var",
        ],
    )
    def test_statistics(self, func):
        df = lt.DataFrame(example_list_dict)
        pdf = pd.DataFrame(example_list_dict)
        assert getattr(df, func)() == getattr(pdf, func)()
        assert getattr(df, func)(axis=0) == getattr(pdf, func)(axis=0)
        assert getattr(df, func)(axis=1) == getattr(pdf, func)(axis=1)
        if func in ["std", "var"]:
            # The behavior of std and var with axis=None is deprecated
            assert getattr(df, func)(axis=None) == getattr(pdf.values.ravel(), func)(
                ddof=1
            )  # Numpy defaults to population std
        else:
            assert getattr(df, func)(axis=None) == getattr(pdf, func)(axis=None)

    def test_statistics_mode(self):
        # @TODO: This is a mess
        example_mode_input = [[0, 1, 1], [2, 2, 1], [3, 3, 2], [0, 0, 2], [1, 2, 1]]
        df = lt.DataFrame(example_mode_input)
        pdf = pd.DataFrame(example_mode_input)
        assert_series_equal_pandas(df.mode(axis=0), pdf.mode(axis=0).T[0].rename(None))
        assert_series_equal_pandas(df.mode(axis=1), pdf.mode(axis=1)[0].rename(None))

    def test_statistics_quantiles(self):
        df = lt.DataFrame(example_list_dict)
        transposed = lt.DataFrame(example_list_dict).T.values
        assert df.quantiles(axis=0).values == [statistics.quantiles(row) for row in transposed]
        assert df.quantiles(axis=1).values == [statistics.quantiles(row.values()) for row in example_list_dict]


class TestDataFrameExports:
    def test_to_list(self):
        df = lt.DataFrame(example_list_dict)
        pdf = pd.DataFrame(example_list_dict)
        assert df.to_list() == pdf.values.tolist()

    def test_to_dict(self):
        df = lt.DataFrame(example_list_dict)
        pdf = pd.DataFrame(example_list_dict)
        assert df.to_dict() == pdf.to_dict()
        assert df.to_dict(orient="dict") == pdf.to_dict(orient="dict")
        assert df.to_dict(orient="list") == pdf.to_dict(orient="list")
        assert df.to_dict(orient="records") == pdf.to_dict(orient="records")
        assert_exception(lambda: pdf.to_dict(orient="error"), lambda: df.to_dict(orient="error"), ValueError)


class TestDataFrameComparisons:
    def test_op_error_non_identically_labeled_dataframes(self):
        dfa = lt.DataFrame(example_list_dict)
        dfb = lt.DataFrame(example_dict_list)
        pdfa = pd.DataFrame(example_list_dict)
        pdfb = pd.DataFrame(example_dict_list)
        assert_exception(lambda: pdfa == pdfb, lambda: dfa == dfb, ValueError)

    def test_op_error_non_empty_dataframe_and_empty_collection(self):
        df = lt.DataFrame(example_dict_list)
        pdf = pd.DataFrame(example_dict_list)
        cmp = []
        assert_exception(lambda: pdf == cmp, lambda: df == cmp, ValueError)

    def test_eq_empty_empty_dataframe(self):
        df = lt.DataFrame() == lt.DataFrame()
        pdf = pd.DataFrame() == pd.DataFrame()
        assert_dataframe_equal_pandas(df, pdf)

    def test_eq_empty_empty_series(self):
        df = lt.DataFrame() == lt.Series()
        pdf = pd.DataFrame() == pd.Series()
        assert_dataframe_equal_pandas(df, pdf)

    def test_eq_empty_empty_collection(self):
        df = lt.DataFrame() == []
        pdf = pd.DataFrame() == []
        assert_dataframe_equal_pandas(df, pdf)

    # def test_eq_empty_empty_2d_collection(self):
    #     # Pandas does create a DataFrame with zero columns and one row
    #     # We're keeping it empty
    #     df = lt.DataFrame() == [[]]
    #     pdf = pd.DataFrame() == [[]]
    #     assert_dataframe_equal_pandas(df, )

    def test_eq_empty_empty_mapping(self):
        df = lt.DataFrame() == {}
        pdf = pd.DataFrame() == {}
        assert_dataframe_equal_pandas(df, pdf)

    def test_eq_empty_scalar(self):
        df = lt.DataFrame() == example_scalar
        pdf = pd.DataFrame() == example_scalar
        assert_dataframe_equal_pandas(df, pdf)

    def test_eq_scalar(self):
        df = lt.DataFrame(example_list_dict) == example_scalar
        pdf = pd.DataFrame(example_list_dict) == example_scalar
        assert_dataframe_equal_pandas(df, pdf)

    def test_eq_empty_collection(self):
        cmp_list = [0, 4]
        assert_exception(lambda: pd.DataFrame() == cmp_list, lambda: lt.DataFrame() == cmp_list, ValueError)

    def test_eq_collection(self):
        cmp_list = [0, 4]
        df = lt.DataFrame(example_list_dict) == cmp_list
        pdf = pd.DataFrame(example_list_dict) == cmp_list
        assert_dataframe_equal_pandas(df, pdf)

    def test_eq_empty_series(self):
        cmp_list = [0, 4]
        assert_exception(
            lambda: pd.DataFrame() == pd.Series(cmp_list), lambda: lt.DataFrame() == lt.Series(cmp_list), ValueError
        )

    def test_eq_series(self):
        cmp_list = [0, 4]
        df = lt.DataFrame(example_list_dict)
        df = df == lt.Series(cmp_list, index=df.columns)
        pdf = pd.DataFrame(example_list_dict)
        pdf = pdf == pd.Series(cmp_list, index=df.columns)
        assert_dataframe_equal_pandas(df, pdf)

    def test_eq_empty_mapping(self):
        cmp_list = {0: 1, 2: 3}
        assert_exception(lambda: pd.DataFrame() == cmp_list, lambda: lt.DataFrame() == cmp_list, ValueError)

    def test_eq_mapping(self):
        cmp_list = {"a": 0, "b": 4}
        df = lt.DataFrame(example_list_dict) == cmp_list
        pdf = pd.DataFrame(example_list_dict) == cmp_list
        assert_dataframe_equal_pandas(df, pdf)

    def test_lt_ge(self):
        dfa = lt.DataFrame(example_cmp_a)
        dfb = lt.DataFrame(example_cmp_b)
        pdfa = pd.DataFrame(example_cmp_a)
        pdfb = pd.DataFrame(example_cmp_b)
        assert_dataframe_equal_pandas(dfa < dfb, pdfa < pdfb)
        assert_dataframe_equal_pandas(dfa >= dfb, pdfa >= pdfb)

    def test_le_gt(self):
        dfa = lt.DataFrame(example_cmp_a)
        dfb = lt.DataFrame(example_cmp_b)
        pdfa = pd.DataFrame(example_cmp_a)
        pdfb = pd.DataFrame(example_cmp_b)
        assert_dataframe_equal_pandas(dfa > dfb, pdfa > pdfb)
        assert_dataframe_equal_pandas(dfa <= dfb, pdfa <= pdfb)

    def test_eq_dataframe(self):
        dfa = lt.DataFrame(example_cmp_a)
        dfb = lt.DataFrame(example_cmp_b)
        pdfa = pd.DataFrame(example_cmp_a)
        pdfb = pd.DataFrame(example_cmp_b)
        assert_dataframe_equal_pandas(dfa == dfb, pdfa == pdfb)

    def test_ne(self):
        dfa = lt.DataFrame(example_cmp_a)
        dfb = lt.DataFrame(example_cmp_b)
        pdfa = pd.DataFrame(example_cmp_a)
        pdfb = pd.DataFrame(example_cmp_b)
        assert_dataframe_equal_pandas(dfa != dfb, pdfa != pdfb)


# class TestDataFrameOperators:
#     @pytest.mark.parametrize(
#         "op",
#         [
#             "__add__",
#             "__sub__",
#             "__mul__",
#             "__truediv__",
#             "__floordiv__",
#             "__mod__",
#             "__pow__",
#             "__radd__",
#             "__rsub__",
#             "__rmul__",
#             "__rtruediv__",
#             "__rfloordiv__",
#             "__rmod__",
#             "__rpow__",
#         ],
#     )
#     def test_op(self, op):
#         sa = lt.DataFrame(example_list_dict_a)
#         sb = lt.DataFrame(example_list_dict_b)
#         psa = pd.DataFrame(example_list_dict_a)
#         psb = pd.DataFrame(example_list_dict_b)

#         # DataFrame
#         assert (getattr(sa, op)(sb) == getattr(psa, op)(psb)).all()
#         # Scalar
#         assert (getattr(sa, op)(example_scalar) == getattr(psa, op)(example_scalar)).all()
#         # Collection
#         assert (getattr(sa, op)(example_values) == getattr(psa, op)(example_values)).all()

#     def test_matmul(self):
#         sa = lt.DataFrame(example_list_dict_a)
#         sb = lt.DataFrame(example_list_dict_b)
#         psa = pd.DataFrame(example_list_dict_a)
#         psb = pd.DataFrame(example_list_dict_b)

#         # DataFrame
#         assert (sa @ sb) == (psa @ psb)
#         # Collection
#         assert (sa @ example_values) == (psa @ example_values)

#     @pytest.mark.parametrize(
#         "op",
#         [
#             "__and__",
#             "__xor__",
#             "__or__",
#             "__rand__",
#             "__rxor__",
#             "__ror__",
#         ],
#     )
#     def test_bop(self, op):
#         # DataFrame
#         sa = lt.DataFrame(example_list_dict_a)
#         sb = lt.DataFrame(example_list_dict_b)
#         psa = pd.DataFrame(example_list_dict_a)
#         psb = pd.DataFrame(example_list_dict_b)
#         assert (getattr(sa, op)(sb) == getattr(psa, op)(psb)).all()

#         # Scalar
#         sa = lt.DataFrame(example_list_dict_a)
#         assert (getattr(sa, op)(example_scalar) == getattr(psa, op)(example_scalar)).all()

#         # Collection
#         sa = lt.DataFrame(example_list_dict_a)
#         assert (getattr(sa, op)(example_values) == getattr(psa, op)(np.array(example_values))).all()
#         # Pandads is deprecating logical ops for dtype-less sequqences (eg: list, tuple)

#     @pytest.mark.parametrize(
#         "op",
#         [
#             "__divmod__",
#             "__lshift__",
#             "__rshift__",
#             "__rdivmod__",
#             "__rlshift__",
#             "__rrshift__",
#         ],
#     )
#     def test_rop(self, op):
#         # DataFrame
#         sa = lt.DataFrame(example_list_dict_a)
#         sb = lt.DataFrame(example_list_dict_b)
#         assert getattr(sa, op)(sb) == {k: getattr(v, op)(example_list_dict_b[k]) for k, v in example_list_dict_a.items()}

#         # Scalar
#         sa = lt.DataFrame(example_list_dict_a)
#         assert getattr(sa, op)(example_scalar) == {k: getattr(v, op)(example_scalar) for k, v in example_list_dict_a.items()}

#         # Collection
#         sa = lt.DataFrame(example_list_dict_a)
#         assert getattr(sa, op)(example_values) == {
#             k: getattr(v, op)(example_values[i]) for i, (k, v) in enumerate(example_list_dict_a.items())
#         }

#     @pytest.mark.parametrize(
#         "iop",
#         [
#             "__iadd__",
#             "__isub__",
#             "__imul__",
#             # "__imatmul__",
#             "__itruediv__",
#             "__ifloordiv__",
#             "__imod__",
#             "__ipow__",
#             "__iand__",
#             "__ixor__",
#             "__ior__",
#         ],
#     )
#     def test_op_inplace(self, iop):
#         sa = lt.DataFrame(example_list_dict_a)
#         sb = lt.DataFrame(example_list_dict_b)
#         psa = pd.DataFrame(example_list_dict_a)
#         psb = pd.DataFrame(example_list_dict_b)
#         getattr(sa, iop)(sb)
#         getattr(psa, iop)(psb)
#         assert (sa == psa).all()

#     @pytest.mark.parametrize(
#         ("iop", "op"),
#         [
#             ("__ilshift__", "__lshift__"),
#             ("__irshift__", "__rshift__"),
#         ],
#     )
#     def test_op_inplace_shift(self, iop, op):
#         sa = lt.DataFrame(example_list_dict_a)
#         sb = lt.DataFrame(example_list_dict_b)
#         getattr(sa, iop)(sb)
#         assert sa == {k: getattr(v, op)(example_list_dict_b[k]) for k, v in example_list_dict_a.items()}

#     def test_different_length_op_error(self):
#         sa = lt.DataFrame(example_list_dict_a)
#         sb = lt.DataFrame(example_list_dict_b)[: len(example_list_dict_b) - 2]
#         with pytest.raises(ValueError, match="Cannot operate"):
#             sa + sb


class TestDataFrameUnaryOperators:
    def test_neg(self):
        df = lt.DataFrame(example_unary)
        pdf = pd.DataFrame(example_unary)
        assert_dataframe_equal_pandas(-df, -pdf)

    def test_pos(self):
        df = lt.DataFrame(example_unary)
        pdf = pd.DataFrame(example_unary)
        assert_dataframe_equal_pandas(+df, +pdf)

    def test_abs(self):
        df = lt.DataFrame(example_unary)
        pdf = pd.DataFrame(example_unary)
        assert_dataframe_equal_pandas(abs(df), abs(pdf))

    def test_invert(self):
        df = lt.DataFrame(example_unary)
        pdf = pd.DataFrame(example_unary)
        assert_dataframe_equal_pandas(~df, ~pdf)
