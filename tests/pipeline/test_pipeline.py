# -*- coding: utf-8 -*-
"""Test for ``pysparkpipe.pipeline.pipeline``"""


import concurrent.futures
import time

import numpy as np
import pandas as pd
import pytest
from pandera import Column, DataFrameSchema
from pyspark.sql import SparkSession
from typeguard import TypeCheckError

from pysparkpipe.exc import (
    InputLayerMissingGroupingColsError,
    LayersConsistencyError,
    PipelineCompileError,
)
from pysparkpipe.pipeline import Pipeline
from pysparkpipe.pipeline.utils import (
    create_spark_dataframe_using_pandera_model,
)


def valid_input_for_layer():
    valid_input = {
        "transform": lambda x: x,
        "grouping_cols": ["col1", "col2"],
        "input_schema": DataFrameSchema(
            {
                "col1": Column(int, coerce=True),
                "col2": Column(float, nullable=True, coerce=True),
                "col3": Column(str, nullable=True, coerce=True),
            }
        ),
        "output_schema": DataFrameSchema(
            {
                "col1": Column(int, coerce=True),
                "col2": Column(float, nullable=True, coerce=True),
                "col3": Column(str, nullable=True, coerce=True),
            }
        ),
        "validate_input": True,
        "validate_output": True,
    }
    return valid_input


def test_invalid_input_errors():
    """
    The class Pipeline should raise an error when instantiated:
        TypeCheckError: If grouping_cols is not a list of strings.
        TypeCheckError: If validate_inputs is not a boolean.
        TypeCheckError: If validate_outputs is not a boolean.
    """
    with pytest.raises(TypeCheckError):
        Pipeline(
            grouping_cols=None, validate_inputs=True, validate_outputs=True
        )

    with pytest.raises(TypeCheckError):
        Pipeline(
            grouping_cols=[1, "col1"],
            validate_inputs=True,
            validate_outputs=True,
        )

    with pytest.raises(TypeCheckError):
        Pipeline(
            grouping_cols=["col1", "col2"],
            validate_inputs=None,
            validate_outputs=True,
        )

    with pytest.raises(TypeCheckError):
        Pipeline(
            grouping_cols=["col1", "col2"],
            validate_inputs=True,
            validate_outputs=None,
        )
    # finally a valid input
    Pipeline(
        grouping_cols=["col1", "col2"],
        validate_inputs=True,
        validate_outputs=True,
    )


def test_add_method_raises_error():
    pipe = Pipeline(
        grouping_cols=["col1", "col2"],
        validate_inputs=True,
        validate_outputs=True,
    )

    with pytest.raises(TypeCheckError):
        # this error should come from Layer
        pipe.add(transform=None, input_schema=None, output_schema=None)

    def transform(x):
        return x

    schema1 = DataFrameSchema(
        {
            "col1": Column(int, coerce=True),
            "col2": Column(float, nullable=True, coerce=True),
            "col3": Column(str, nullable=True, coerce=True),
        }
    )
    schema2_good = DataFrameSchema(
        {
            "col1": Column(int, coerce=True),
            "col2": Column(float, nullable=True, coerce=True),
            "col3": Column(str, nullable=True, coerce=True),
            "col4": Column(str, nullable=True, coerce=True),
        }
    )
    schema2_bad = DataFrameSchema(  # this doesnt contain all grouping cols
        {
            "col1": Column(int, coerce=True),
            "col3": Column(float, nullable=True, coerce=True),
            "col4": Column(str, nullable=True, coerce=True),
        }
    )
    schema3 = DataFrameSchema(
        {
            "col1": Column(int, coerce=True),
            "col2": Column(float, nullable=True, coerce=True),
        }
    )

    # this error should come from Layer, missing grouping cols in the input layer
    pipe = Pipeline(
        grouping_cols=["col1", "col2"],
        validate_inputs=True,
        validate_outputs=True,
    )
    with pytest.raises(InputLayerMissingGroupingColsError):
        pipe.add(transform, schema2_bad, schema1)

    # this error should come from Pipeline, inconsistent Layers
    pipe = Pipeline(
        grouping_cols=["col1", "col2"],
        validate_inputs=True,
        validate_outputs=True,
    )
    with pytest.raises(LayersConsistencyError):
        pipe.add(transform, schema1, schema2_good)
        pipe.add(transform, schema3, schema3)

    # this should work
    pipe = Pipeline(
        grouping_cols=["col1", "col2"],
        validate_inputs=True,
        validate_outputs=True,
    )
    # finally a valid input
    pipe.add(transform, schema1, schema2_good)
    pipe.add(transform, schema2_good, schema3)


def test_input_and_output_schemas():
    pipe = Pipeline(
        grouping_cols=["col1", "col2"],
        validate_inputs=True,
        validate_outputs=True,
    )

    schema1 = DataFrameSchema(
        {
            "col1": Column(int, coerce=True),
            "col2": Column(float, nullable=True, coerce=True),
            "col3": Column(str, nullable=True, coerce=True),
        }
    )
    schema2 = DataFrameSchema(
        {
            "col1": Column(int, coerce=True),
            "col2": Column(float, nullable=True, coerce=True),
            "col3": Column(str, nullable=True, coerce=True),
            "col4": Column(str, nullable=True, coerce=True),
        }
    )

    schema3 = DataFrameSchema(
        {
            "col1": Column(int, coerce=True),
            "col2": Column(float, nullable=True, coerce=True),
        }
    )

    # initially pipe.layers is empty
    assert pipe.input_schema is None
    assert pipe.output_schema is None

    pipe.add(lambda x: x, schema1, schema2)
    for _ in range(10):
        pipe.add(lambda x: x, schema2, schema2)
    pipe.add(lambda x: x, schema2, schema3)

    assert pipe.input_schema == schema1
    assert pipe.output_schema == schema3


def test_pipeline_fit():
    pipe = Pipeline(
        grouping_cols=["col1"],
        validate_inputs=True,
        validate_outputs=True,
    )

    schema = DataFrameSchema(
        {
            "col1": Column(float, nullable=False, coerce=True),
        }
    )

    # input data
    df = pd.DataFrame(data={"col1": ["2"]})

    # expected data after the pipeline
    df_expected = pd.DataFrame(data={"col1": [2.0**11.0]})

    # transformation applied in each layer
    def transform(x):
        x["col1"] = x["col1"] * 2.0
        return x

    for _ in range(10):
        pipe.add(transform, schema, schema)

    df_output = pipe.fit(df)

    pd.testing.assert_frame_equal(df_output, df_expected)


def test_pipeline_apply_in_pandas():
    pipe = Pipeline(
        grouping_cols=["col1"],
        validate_inputs=True,
        validate_outputs=True,
    )

    schema = DataFrameSchema(
        {
            "col1": Column(str, nullable=False, coerce=True),
            "col2": Column(float, nullable=False, coerce=True),
        }
    )

    # input data
    df = pd.DataFrame(
        data={"col1": [1, 1, "2", "2", "3", "3"], "col2": [0, 1, 0, 2, 0, 3]}
    )

    # expected data after the pipeline
    df_expected = pd.DataFrame(
        data={
            "col1": ["1", "2", "3"],
            "col2": (2.0**10.0) * np.asarray([1.0, 2.0, 3.0]),
        }
    )

    # transformation applied in each layer
    def transform_max(x):
        col1 = [x["col1"].iloc[0]]
        col2 = [np.max(x["col2"])]
        return pd.DataFrame({"col1": col1, "col2": col2})

    def transform_mult(x):
        x["col2"] = x["col2"] * 2.0
        return x

    pipe.add(transform_max, schema, schema)
    for _ in range(10):
        pipe.add(transform_mult, schema, schema)

    df_output = pipe.apply_in_pandas(df).reset_index(drop=True)

    pd.testing.assert_frame_equal(df_output, df_expected)


def test_pipeline_apply_in_spark(spark: SparkSession = None):
    if not spark:
        spark = (
            SparkSession.builder.master("local[2]")
            .appName("pysparkpipe-test")
            .getOrCreate()
        )

    pipe = Pipeline(
        grouping_cols=["col1"],
        validate_inputs=True,
        validate_outputs=True,
    )

    schema = DataFrameSchema(
        {
            "col1": Column(str, nullable=False, coerce=True),
            "col2": Column(float, nullable=False, coerce=True),
        }
    )

    # input data
    df = pd.DataFrame(
        data={"col1": [1, 1, "2", "2", "3", "3"], "col2": [0, 1, 0, 2, 0, 3]}
    )
    df = create_spark_dataframe_using_pandera_model(schema, spark, df)

    # expected data after the pipeline
    df_expected = pd.DataFrame(
        data={
            "col1": ["1", "2", "3"],
            "col2": (2.0**10.0) * np.asarray([1.0, 2.0, 3.0]),
        }
    )

    # transformation applied in each layer
    def transform_max(x):
        col1 = [x["col1"].iloc[0]]
        col2 = [np.max(x["col2"])]
        return pd.DataFrame({"col1": col1, "col2": col2})

    def transform_mult(x):
        x["col2"] = x["col2"] * 2.0
        return x

    pipe.add(transform_max, schema, schema)
    for _ in range(10):
        pipe.add(transform_mult, schema, schema)

    df_output = pipe.apply_in_spark(df).toPandas()

    pd.testing.assert_frame_equal(df_output, df_expected)


def test_cogroup_apply_in_spark(spark: SparkSession = None):
    if not spark:
        spark = (
            SparkSession.builder.master("local[2]")
            .appName("pysparkpipe-test")
            .getOrCreate()
        )

    pipe = Pipeline(
        grouping_cols=["col1"],
        validate_inputs=True,
        validate_outputs=True,
    )

    schema1 = DataFrameSchema(
        {
            "col1": Column(int, nullable=False, coerce=True),
            "col2": Column(str, nullable=False, coerce=True),
        }
    )

    schema2 = DataFrameSchema(
        {
            "col1": Column(int, nullable=False, coerce=True),
            "col3": Column(str, nullable=False, coerce=True),
        }
    )

    # input data
    df1 = pd.DataFrame(
        data={"col1": [1, 1, "2", "2", "3", "3"], "col2": [0, 1, 0, 2, 0, 3]}
    )
    df1 = create_spark_dataframe_using_pandera_model(schema1, spark, df1)

    df2 = pd.DataFrame(
        data={"col1": [1.0, 1.0, "2", "2", 3.0, "3"], "col3": "foo"}
    )
    df2 = create_spark_dataframe_using_pandera_model(schema2, spark, df2)

    # expected data after the pipeline
    df_expected = pd.DataFrame(
        data={
            "col1": [1, 2, 3],
            "col2": ["1", "2", "3"],
            "col3": ["foo", "foo", "foo"],
        }
    )

    # output schema
    schema_out = DataFrameSchema(
        {
            "col1": Column(int, nullable=False, coerce=True),
            "col2": Column(str, nullable=False, coerce=True),
            "col3": Column(str, nullable=False, coerce=True),
        }
    )

    def transform_join(df1, df2):
        return df1.merge(df2, on="col1", how="outer")

    def transform_max(x):
        col1 = [x["col1"].iloc[0]]
        col2 = [np.max(x["col2"])]
        col3 = [x["col3"].iloc[0]]
        return pd.DataFrame({"col1": col1, "col2": col2, "col3": col3})

    pipe.add(transform_join, [schema1, schema2], schema_out)

    pipe.add(transform_max, schema_out, schema_out)

    df_output = pipe.apply_in_spark(df=df1, df2=df2).toPandas()

    pd.testing.assert_frame_equal(df_output, df_expected)

    # assert that the pipeline fails if the input is not a spark dataframe
    with pytest.raises(PipelineCompileError):
        pipe.apply_in_spark(df=df1, df2=None)


def test_pipeline_raises_no_exception_handler():
    # case 1: exception handler is not passed
    pipe = Pipeline(
        grouping_cols=["col1"],
        validate_inputs=True,
        validate_outputs=True,
    )

    schema = DataFrameSchema(
        {
            "col1": Column(float, nullable=False, coerce=True),
        }
    )

    # input data
    df = pd.DataFrame(data={"col1": ["2"]})

    # transformation applied in each layer
    def transform(x):
        x["col1"] = x["col1"] * 2.0
        if x["col1"].iloc[0] == 2.0**11.0:
            raise ValueError("This is a test exception")
        return x

    for _ in range(10):
        pipe.add(transform, schema, schema)

    # assert that the pipeline raises the exception, since no exception handler is passed
    with pytest.raises(ValueError):
        pipe.fit(df)


def test_pipeline_exception_handler():
    # case 2: exception handler passed

    schema = DataFrameSchema(
        {
            "col1": Column(float, nullable=True, coerce=True),
            "msg": Column(str, nullable=True, coerce=True),
        }
    )

    # expected data after the pipeline
    df_expected = pd.DataFrame(
        data={"col1": [np.nan], "msg": ["This is a test exception"]}
    )

    # define the exception handler
    def exception_handler(e, df):
        df["col1"] = np.nan
        df["msg"] = str(e)
        return df

    pipe = Pipeline(
        grouping_cols=["col1"],
        validate_inputs=True,
        validate_outputs=True,
        exception_handler=exception_handler,
    )

    # input data
    df = pd.DataFrame(data={"col1": ["2"]})

    # transformation applied in each layer
    def transform(x):
        x["col1"] = x["col1"] * 2.0
        if x["col1"].iloc[0] == 2.0**11.0:
            raise ValueError("This is a test exception")
        return x

    for _ in range(10):
        pipe.add(transform, schema, schema)

    # assert that output is as expected
    df_output = pipe.fit(df)

    pd.testing.assert_frame_equal(df_output, df_expected)


@pytest.mark.parametrize(
    "timeout, df_expected",
    [
        (None, pd.DataFrame(data={"col1": [2.0], "msg": ["Ok"]})),
        (1, pd.DataFrame(data={"col1": [2.0], "msg": ["TimeoutError"]})),
    ],
)
def test_pipeline_exception_handler_with_timeout(timeout, df_expected):
    # case 3: exception handler passed with a timeout {None, 1}
    sleep_time = 1.25
    schema = DataFrameSchema(
        {
            "col1": Column(float, nullable=True, coerce=True),
            "msg": Column(str, nullable=True, coerce=True),
        }
    )

    # define the exception handler
    def exception_handler(e, df):
        if isinstance(e, concurrent.futures._base.TimeoutError):
            df["msg"] = "TimeoutError"
        else:
            df["msg"] = str(e)
        return df

    pipe = Pipeline(
        grouping_cols=["col1"],
        validate_inputs=True,
        validate_outputs=True,
        exception_handler=exception_handler,
        timeout=timeout,
    )

    # input data
    df = pd.DataFrame(data={"col1": ["2"]})

    # transformation applied in each layer
    def transform(x):
        # this will raise a timeout error depending on the timeout parameter
        time.sleep(sleep_time)
        x["msg"] = "Ok"
        return x

    # add the transform layer
    pipe.add(transform, schema, schema)

    # assert that output is as expected
    df_output = pipe.fit(df)

    pd.testing.assert_frame_equal(df_output, df_expected)


def test_pipeline_raises_timeout_error():
    # case 4: there is no exception handler and the timeout is reached

    schema = DataFrameSchema(
        {
            "col1": Column(float, nullable=True, coerce=True),
            "msg": Column(str, nullable=True, coerce=True),
        }
    )

    pipe = Pipeline(
        grouping_cols=["col1"],
        validate_inputs=True,
        validate_outputs=True,
        exception_handler=None,
        timeout=1,
    )

    # input data
    df = pd.DataFrame(data={"col1": ["2"]})

    # transformation applied in each layer
    def transform(x):
        # this will raise a timeout error
        time.sleep(2)
        return x

    # add the transform layer
    pipe.add(transform, schema, schema)

    # assert that the error is raised
    pytest.raises(concurrent.futures._base.TimeoutError, pipe.fit, df)


def test_pipeline_exception_handler_in_apply_in_spark(
    spark: SparkSession = None,
):
    if not spark:
        spark = (
            SparkSession.builder.master("local[2]")
            .appName("pysparkpipe-test")
            .getOrCreate()
        )

    schema_in = DataFrameSchema(
        {
            "col1": Column(str, nullable=False, coerce=True),
            "col2": Column(float, nullable=False, coerce=True),
        }
    )
    schema_out = DataFrameSchema(
        {
            "col1": Column(str, nullable=False, coerce=True),
            "col2": Column(float, nullable=True, coerce=True),
            "msg": Column(str, nullable=True, coerce=True),
        }
    )

    # input data
    df = pd.DataFrame(
        data={"col1": [1, 1, "2", "2", "3", "3"], "col2": [0, 1, 0, 2, 0, 3]}
    )
    df = create_spark_dataframe_using_pandera_model(schema_in, spark, df)

    # expected data after the pipeline
    df_expected = pd.DataFrame(
        data={
            "col1": ["1", "2", "3"],
            "col2": np.asarray([2.0**10.0, np.nan, 3.0 * 2.0**10.0]),
            "msg": [
                "some message",
                "This is a test exception",
                "some message",
            ],
        }
    )

    # transformation applied in each layer
    def transform_max(x):
        col1 = [x["col1"].iloc[0]]
        col2 = [np.max(x["col2"])]
        return pd.DataFrame({"col1": col1, "col2": col2})

    def transform_mult(x):
        x["col2"] = x["col2"] * 2.0
        return x

    def transform_add_msg(x):
        x["msg"] = "some message"
        if x["col1"].iloc[0] == "2":
            raise ValueError("This is a test exception")
        return x

    # define the exception handler
    def exception_handler(e, df):
        df_exc = pd.DataFrame(
            {"col1": [df["col1"].iloc[0]], "col2": [np.nan], "msg": [str(e)]}
        )
        return df_exc

    pipe = Pipeline(
        grouping_cols=["col1"],
        validate_inputs=True,
        validate_outputs=True,
        exception_handler=exception_handler,
    )

    # add layers
    pipe.add(transform_max, schema_in, schema_in)
    for _ in range(10):
        pipe.add(transform_mult, schema_in, schema_in)
    pipe.add(transform_add_msg, schema_in, schema_out)

    # compute output
    df_output = pipe.apply_in_spark(df).toPandas()

    # assert that output is as expected
    pd.testing.assert_frame_equal(df_output, df_expected)


def test_pipeline_exception_handler_in_apply_in_spark_timeout(
    spark: SparkSession = None,
):
    sleep_time = 1.25

    if not spark:
        spark = (
            SparkSession.builder.master("local[2]")
            .appName("pysparkpipe-test")
            .getOrCreate()
        )

    schema_in = DataFrameSchema(
        {
            "col1": Column(str, nullable=False, coerce=True),
            "col2": Column(float, nullable=False, coerce=True),
        }
    )
    schema_out = DataFrameSchema(
        {
            "col1": Column(str, nullable=False, coerce=True),
            "col2": Column(float, nullable=True, coerce=True),
            "msg": Column(str, nullable=True, coerce=True),
        }
    )

    # input data
    df = pd.DataFrame(
        data={
            "col1": [1, 1, "2", "2", "3", "3", "4", "4"],
            "col2": [0, 1, 0, 2, 0, 3, 4, 4],
        }
    )
    df = create_spark_dataframe_using_pandera_model(schema_in, spark, df)

    # expected data after the pipeline
    df_expected = pd.DataFrame(
        data={
            "col1": ["1", "2", "3", "4"],
            "col2": np.asarray(
                [2.0**10.0, np.nan, 3.0 * 2.0**10.0, np.nan]
            ),
            "msg": [
                "some message",
                "This is a test exception",
                "some message",
                "TimeoutError",
            ],
        }
    )

    # transformation applied in each layer
    def transform_max(x):
        col1 = [x["col1"].iloc[0]]
        col2 = [np.max(x["col2"])]
        return pd.DataFrame({"col1": col1, "col2": col2})

    def transform_mult(x):
        x["col2"] = x["col2"] * 2.0
        return x

    def transform_add_msg(x):
        x["msg"] = "some message"
        if x["col1"].iloc[0] == "2":
            raise ValueError("This is a test exception")
        return x

    def transform_sleep(x):
        if x["col1"].iloc[0] == "4":
            time.sleep(sleep_time)
        return x

    # define the exception handler
    def exception_handler(e, df):
        df_exc = pd.DataFrame(
            {"col1": [df["col1"].iloc[0]], "col2": [np.nan], "msg": [str(e)]}
        )
        if isinstance(e, concurrent.futures._base.TimeoutError):
            df_exc["msg"] = "TimeoutError"
        return df_exc

    pipe = Pipeline(
        grouping_cols=["col1"],
        validate_inputs=True,
        validate_outputs=True,
        exception_handler=exception_handler,
        timeout=1,
    )

    # add layers
    pipe.add(transform_max, schema_in, schema_in)
    for _ in range(10):
        pipe.add(transform_mult, schema_in, schema_in)
    pipe.add(transform_add_msg, schema_in, schema_out)
    pipe.add(transform_sleep, schema_out, schema_out)

    # compute output
    df_output = pipe.apply_in_spark(df).toPandas()

    # assert that output is as expected
    pd.testing.assert_frame_equal(df_output, df_expected)


def test_pipeline_repr():
    pipe = Pipeline(
        grouping_cols=["col1"],
        validate_inputs=True,
        validate_outputs=True,
    )

    schema = DataFrameSchema(
        {
            "col1": Column(float, nullable=False, coerce=True),
        }
    )

    def transform(x):
        return x

    pipe.add(transform, schema, schema)

    representation_lines_fragments = [
        "PySparkPipe Pipeline",
        "Layer: transform",
        "Input Schema:",
        "Output Schema:",
        "Transform: <bound method Layer.transform",
        "Validate Input: True",
        "Validate Output: True",
    ]

    representation = pipe.__repr__()

    for fragment in representation_lines_fragments:
        assert fragment in representation
