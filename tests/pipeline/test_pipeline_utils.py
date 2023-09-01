# -*- coding: utf-8 -*-
"""Test for ``pysparkpipe.pipeline.utils``"""


import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from pandera import Column, DataFrameSchema
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from pysparkpipe.pipeline.utils import (
    create_spark_dataframe_using_pandera_model,
    pandera_model_to_spark_structype,
    parse_dataframe_from_dict,
    parse_dataframe_using_pandera_model,
)


def test_pandera_model_to_spark_structype():
    """Test ``pandera_model_to_spark_structype``."""

    # define a pandera model
    pandera_model = DataFrameSchema(
        {
            "column1": Column(int, coerce=True),
            "column2": Column(float, nullable=True, coerce=True),
            # you can provide a list of validators
            "column3": Column(str, nullable=True, coerce=True),
        }
    )

    # define the expected spark structype
    spark_structype_expected = StructType(
        [
            StructField("column1", LongType(), False),
            StructField("column2", DoubleType(), True),
            StructField("column3", StringType(), True),
        ]
    )

    # convert the pandera model to spark structype
    spark_structype = pandera_model_to_spark_structype(pandera_model)

    # check that the spark structype is correct
    assert spark_structype == spark_structype_expected


def test_parse_dataframe_using_pandera_model():
    """Test ``parse_dataframe_using_pandera_model``."""

    # define a pandera model
    pandera_model = DataFrameSchema(
        {
            "column1": Column(int, coerce=True),
            "column2": Column(float, nullable=True, coerce=True),
            # you can provide a list of validators
            "column3": Column(str, nullable=True, coerce=True),
        }
    )

    # incomplete dataframe
    data_input = {"column1": [5.0, 1.0, 0.0]}
    df = pd.DataFrame(data_input)

    # parse dataframe, should include all columns
    df_parsed = parse_dataframe_using_pandera_model(pandera_model, df)

    data_test = {
        "column1": [5, 1, 0],
        "column2": pd.Series([np.nan, np.nan, np.nan], dtype="float"),
        "column3": pd.Series([np.nan, np.nan, np.nan], dtype="str"),
    }
    # check that the dataframe is parsed
    df_test = pd.DataFrame(data_test)

    assert_frame_equal(df_parsed, df_test)


def test_parse_dataframe_using_pandera_model_in_regex_schema():
    """Test ``parse_dataframe_using_pandera_model`` in the case of regex columns."""

    # define a pandera model
    pandera_model = DataFrameSchema(
        {
            "column1": Column(int, coerce=True),
            "column2": Column(float, nullable=True, coerce=True),
            # you can provide a list of validators
            "column3": Column(str, nullable=True, coerce=True),
            # regex columns
            "value_.*": Column(float, nullable=True, coerce=True, regex=True),
        }
    )

    # incomplete dataframe
    data_input = {"column1": [5.0, 1.0, 0.0], "value_1": [5.0, 1.0, 0.0]}
    df = pd.DataFrame(data_input)

    # parse dataframe, should include all columns but not the regex pattern
    df_parsed = parse_dataframe_using_pandera_model(pandera_model, df)

    data_expected = {
        "column1": [5, 1, 0],
        "column2": pd.Series([np.nan, np.nan, np.nan], dtype="float"),
        "column3": pd.Series([np.nan, np.nan, np.nan], dtype="str"),
        "value_1": [5.0, 1.0, 0.0],
    }

    # check that the dataframe is parsed correctly
    assert_frame_equal(df_parsed, pd.DataFrame(data_expected), check_like=True)


def test_parse_dataframe_from_dict():
    """Test ``parse_dataframe_using_pandera_model``."""

    # define a pandera model
    pandera_model = DataFrameSchema(
        {
            "column1": Column(int, coerce=True),
            "column2": Column(float, nullable=True, coerce=True),
            # you can provide a list of validators
            "column3": Column(str, nullable=True, coerce=True),
        }
    )

    # incomplete dataframe
    data_input = {"column1": [5.0, 1.0, 0.0]}

    data_test = {
        "column1": [5, 1, 0],
        "column2": pd.Series([np.nan, np.nan, np.nan], dtype="float"),
        "column3": pd.Series([np.nan, np.nan, np.nan], dtype="str"),
    }

    # check that the dataframe is parsed
    df_test = pd.DataFrame(data_test)

    # repeat the test with parse_dataframe_from_dict
    df_parsed = parse_dataframe_from_dict(pandera_model, data_input)
    assert_frame_equal(df_parsed, df_test)


def test_create_spark_dataframe_using_pandera_model(
    spark: SparkSession = None,
):
    """Test ``create_spark_dataframe_using_pandera_model``."""

    if not spark:
        spark = (
            SparkSession.builder.master("local[2]")
            .appName("pysparkpipe-test")
            .getOrCreate()
        )

    # define a pandera model
    pandera_model = DataFrameSchema(
        {
            "column1": Column(int, coerce=True),
            "column2": Column(float, nullable=True, coerce=True),
            # you can provide a list of validators
            "column3": Column(str, nullable=False, coerce=True),
        }
    )

    # incomplete dataframe
    data_input = {
        "column1": [5.0, 1.0, 0.0],
        "column2": [5, None, 0],
        "column3": ["a", "b", 1],
    }

    data_test = {
        "column1": [5, 1, 0],
        "column2": pd.Series([5.0, np.nan, 0.0], dtype="float"),
        "column3": pd.Series(["a", "b", "1"], dtype="str"),
    }

    # check that the dataframe is parsed
    df_test = pd.DataFrame(data_test)

    # repeat the test with parse_dataframe_from_dict
    df_parsed = create_spark_dataframe_using_pandera_model(
        pandera_model, spark, data_input
    ).toPandas()
    assert_frame_equal(df_parsed, df_test)
