# -*- coding: utf-8 -*-
"""Test for ``pysparkpipe.pipeline.layers``"""


import pandas as pd
import pytest
from pandera import Column, DataFrameSchema
from pandera.errors import SchemaError
from typeguard import TypeCheckError

from pysparkpipe.pipeline.layers import Layer


def valid_input_for_layer():
    valid_input = {
        "transform": lambda x: x,
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
    The class Layer should raise an error when instantiated:
        TypeCheckError: If transform is not callable.
        TypeCheckError: If input_schema is not instance of DataFrameSchema.
        TypeCheckError: If output_schema is not instance of DataFrameSchema.
        TypeCheckError: If grouping_cols is not subset of input_schema columns.
        TypeCheckError: If grouping_cols is not subset of output_schema columns.
        TypeCheckError: If validate_input is not bool.
        TypeCheckError: If validate_output is not bool.
    """
    with pytest.raises(TypeCheckError):
        invalid_input = valid_input_for_layer()
        invalid_input["transform"] = None
        Layer(**invalid_input)

    with pytest.raises(TypeCheckError):
        invalid_input = valid_input_for_layer()
        invalid_input["input_schema"] = None
        Layer(**invalid_input)

    with pytest.raises(TypeCheckError):
        invalid_input = valid_input_for_layer()
        invalid_input["output_schema"] = None
        Layer(**invalid_input)

    with pytest.raises(TypeCheckError):
        invalid_input = valid_input_for_layer()
        invalid_input["validate_input"] = None
        Layer(**invalid_input)

    with pytest.raises(TypeCheckError):
        invalid_input = valid_input_for_layer()
        invalid_input["validate_output"] = None
        Layer(**invalid_input)

    # finally, test that valid input does not raise an error
    valid_input = valid_input_for_layer()
    Layer(**valid_input)


def test_layers_validate():
    input_schema = DataFrameSchema(
        {
            "col1": Column(int, coerce=True),
            "col2": Column(
                float, nullable=True, coerce=False
            ),  # since coerce=False, this should raise an error
            "col3": Column(str, nullable=True, coerce=False),
        }
    )

    df = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": [1.0, 2, 3.0],
            "col3": ["a", "b", 1],
        }
    )
    layer_kwargs = valid_input_for_layer()
    layer_kwargs["input_schema"] = input_schema
    layer = Layer(**layer_kwargs)

    # test that this data will raise an error
    with pytest.raises(SchemaError):
        layer.validate(df, layer.input_schema)


def test_layers_transform():
    df_input = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": [1.0, 2, 3.0],
            "col3": ["a", "b", 1],
        }
    )
    layer_kwargs = valid_input_for_layer()
    layer = Layer(**layer_kwargs)

    df_expected = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": [1.0, 2.0, 3.0],
            "col3": ["a", "b", "1"],
        }
    )
    df_output = layer.transform(df_input)

    # since the transform function is identity, the output should be the same as the input but with the correct dtypes
    pd.testing.assert_frame_equal(df_output, df_expected)

    def transform(_df):
        _df["col2"] = _df["col2"] ** 2
        return _df

    layer_kwargs = valid_input_for_layer()
    layer_kwargs["transform"] = transform
    layer = Layer(**layer_kwargs)

    df_expected_2 = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": [1.0, 4.0, 9.0],
            "col3": ["a", "b", "1"],
        }
    )
    df_output = layer.transform(df_input)

    pd.testing.assert_frame_equal(df_output, df_expected_2)


def test_layers_concatenation_consistency():
    layer1 = Layer(**valid_input_for_layer())
    layer2 = Layer(**valid_input_for_layer())

    assert layer1.is_consistent_with(
        layer2
    )  # the output schema of layer1 is the same as the input schema of layer2
    assert layer2.is_consistent_with(
        layer1
    )  # the output schema of layer2 is the same as the input schema of layer1

    layer2_kwargs = valid_input_for_layer()
    layer2_kwargs["input_schema"] = DataFrameSchema(
        {
            "col1": Column(int, coerce=True),
            "col2": Column(float, nullable=True, coerce=True),
            "col3": Column(str, nullable=True, coerce=True),
            "col4": Column(str, nullable=True, coerce=True),
        }
    )
    layer2 = Layer(**layer2_kwargs)

    assert layer1.is_consistent_with(
        layer2
    )  # the output schema of layer2 is the same as the input schema of layer1
    assert not layer2.is_consistent_with(
        layer1
    )  # the output schema of layer1 is not the same as the input schema of layer2
