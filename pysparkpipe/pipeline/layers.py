# -*- coding: utf-8 -*-
"""Layer class for the pipeline."""

from typing import List, Union

from pandas import DataFrame as PandasDataFrame
from pandera import DataFrameSchema
from typeguard import TypeCheckError, typechecked

from .utils import parse_dataframe_using_pandera_model


class Layer:
    """Layer class for the pipeline.

    Args:
        transform (callable): Transform function.
        grouping_cols (List[str]): List of columns to group by.
        input_schema (Union[DataFrameSchema, List[DataFrameSchema]]): Input schema(s), must contain grouping_cols.
        output_schema (Union[DataFrameSchema, List[DataFrameSchema]]): Output schema(s), must contain grouping_cols.
        validate_input (bool, optional): Validate input schema. Defaults to True.
        validate_output (bool, optional): Validate output schema. Defaults to True.

    Returns:
        None

    Raises:
        TypeError: If transform is not callable.
        TypeError: If input_schema is not instance of DataFrameSchema.
        TypeError: If output_schema is not instance of DataFrameSchema.
        TypeError: If grouping_cols is not subset of input_schema columns.
        TypeError: If grouping_cols is not subset of output_schema columns.
        TypeError: If validate_input is not bool.
        TypeError: If validate_output is not bool.

    """

    @typechecked
    def __init__(
        self,
        transform: callable,
        input_schema: Union[DataFrameSchema, List[DataFrameSchema]],
        output_schema: Union[DataFrameSchema, List[DataFrameSchema]],
        validate_input: bool = True,
        validate_output: bool = True,
    ) -> None:
        """Initialize the layer."""
        self._transform = transform
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.validate_input = validate_input
        self.validate_output = validate_output

        # assert that transform is callable
        if not callable(self._transform):
            raise TypeCheckError("Transform function must be callable.")

        self.name = self._transform.__name__  # type: str

    def validate(
        self,
        df: Union[PandasDataFrame, List[PandasDataFrame]],
        schema: Union[DataFrameSchema, List[DataFrameSchema]],
    ) -> Union[PandasDataFrame, List[PandasDataFrame]]:
        """Validate a pandas DataFrame (or a list of pandas DataFrames) against schema (or list of schemas.)

        Args:
            df (Union[PandasDataFrame, List[PandasDataFrame]]): DataFrame to validate.
            schema (Union[DataFrameSchema, List[DataFrameSchema]]): Schema to validate against.

        Returns:
            Union[PandasDataFrame, List[PandasDataFrame]]: Validated DataFrame (or list of validated DataFrames).
        """
        if isinstance(df, list):
            # if df is a list of dataframes, validate each dataframe and return list of validated dataframes
            return [self.validate(d, s) for d, s in zip(df, schema)]
        return parse_dataframe_using_pandera_model(schema, df)

    def transform(
        self, df: Union[PandasDataFrame, List[PandasDataFrame]]
    ) -> Union[PandasDataFrame, List[PandasDataFrame]]:
        """Transform a pandas DataFrame.

        Args:
            df (DataFrame): DataFrame to transform.

        Returns:
            DataFrame: Transformed DataFrame.
        """
        if self.validate_input:
            # validate input dataframe against input schema
            df = self.validate(df=df, schema=self.input_schema)

        if isinstance(df, list):
            # if df is a list of dataframes, unpack list
            df = self._transform(*df)
        else:
            df = self._transform(df)

        if self.validate_output:
            # validate output dataframe against output schema
            df = self.validate(df=df, schema=self.output_schema)
        return df

    def is_consistent_with(self, other: "Layer") -> bool:
        """Check if layer is consistent with other layer, i.e. if they can be chained in a pipeline (the new layer is applied after the other layer).

        Args:
            other (Layer): Other (previous) layer to check consistency with.

        Returns:
            bool: True if consistent, False otherwise.
        """
        if self.input_schema != other.output_schema:
            return False
        return True
