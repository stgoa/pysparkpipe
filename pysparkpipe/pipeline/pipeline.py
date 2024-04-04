# -*- coding: utf-8 -*-
"""Spark applyInPandas Pipeline class"""

import concurrent.futures
import logging
from typing import List, Optional, Union

from pandas import DataFrame as PandasDataFrame
from pandera import DataFrameSchema
from pyspark.sql import DataFrame as SparkDataFrame
from typeguard import typechecked

from pysparkpipe.exc import (
    InputLayerMissingGroupingColsError,
    LayersConsistencyError,
    PipelineCompileError,
)
from pysparkpipe.pipeline.layers import Layer
from pysparkpipe.pipeline.utils import (
    pandera_model_to_spark_structype,
    parse_dataframe_using_pandera_model,
)

logger = logging.getLogger(__name__)


class Pipeline:
    """Pipeline class. A pipeline is a sequence of layers that are applied to the data grouped by the grouping columns."""

    @typechecked
    def __init__(
        self,
        grouping_cols: List[str],
        validate_inputs: bool = True,
        validate_outputs: bool = True,
        exception_handler: Optional[callable] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """Pipeline constructor.

        Args:
            grouping_cols (List[str]): List of grouping columns.
            validate_inputs (bool, optional): If True, by default the inner Layers will validate the input dataframes. Defaults to True.
            validate_outputs (bool, optional): If True, by defualt the inner Layers will validate the output dataframes. Defaults to True.
            exception_handler (callable, optional): Exception handler to be used by the pipeline. Defaults to None. If passed, the pipeline
                will call the exception_handler with the dataframe and the exception as arguments. The exception_handler should return a dataframe.
                with the same schema as the output schema of the pipeline. If the exception_handler is not passed, the pipeline will raise the exception.
                NOTE: exception_handler(e: Exception, df: PandasDataFrame) -> PandasDataFrame will not be typechecked.
            timeout (int, optional): Timeout in seconds for the pipeline to run. Defaults to None.

        Raises:
            ValueError: If grouping_cols is not a list of strings.
            ValueError: If validate_inputs is not a boolean.
            ValueError: If validate_outputs is not a boolean.
        """

        self.grouping_cols = grouping_cols  # type: List[str]
        self.layers = []  # type: List[Layer]
        self.validate_inputs = validate_inputs  # type: bool
        self.validate_outputs = validate_outputs  # type: bool
        self._exception_handler = exception_handler  # type: Optional[callable]
        self.timeout = timeout  # type: Optional[int]

    def add_layer(
        self,
        new_layer: Layer,
    ):
        # assert new_layer is consistent with previous layers
        if len(self.layers) > 0:
            if not new_layer.is_consistent_with(self.layers[-1]):
                raise LayersConsistencyError(
                    new_layer_name=new_layer.name,
                    prev_layer_name=self.layers[-1].name,
                )
        else:
            if not isinstance(new_layer.input_schema, list):
                sch_lst = [new_layer.input_schema]
            else:
                sch_lst = new_layer.input_schema
            for s in sch_lst:
                if not set(self.grouping_cols).issubset(set(s.columns.keys())):
                    raise InputLayerMissingGroupingColsError(
                        grouping_cols=self.grouping_cols, schema=s
                    )
        # if consistent, add the new layer to pipeline
        self.layers.append(new_layer)

    def add(
        self,
        transform: callable,
        input_schema: DataFrameSchema,
        output_schema: DataFrameSchema,
        validate_input: Optional[bool] = None,
        validate_output: Optional[bool] = None,
    ) -> None:
        # if validate_input is None, use the pipeline's validate_inputs
        if validate_input is None:
            validate_input = self.validate_inputs
        # if validate_output is None, use the pipeline's validate_outputs
        if validate_output is None:
            validate_output = self.validate_outputs

        # create a new layer
        new_layer = Layer(
            transform=transform,
            input_schema=input_schema,
            output_schema=output_schema,
            validate_input=validate_input,
            validate_output=validate_output,
        )
        # add the new layer to pipeline
        self.add_layer(new_layer)

    def compile(
        self,
        df: Optional[Union[SparkDataFrame, PandasDataFrame]] = None,
        df2: Optional[SparkDataFrame] = None,
    ) -> None:
        """Compile the pipeline."""

        if df2 is not None and len(self.input_schema) != 2:
            raise PipelineCompileError(
                msg="Pipeline must have exactly two input schemas to be compiled with two input dataframes."
            )
        elif df2 is None and isinstance(self.input_schema, list):
            raise PipelineCompileError(
                msg="Pipeline must have exactly one input schema to be compiled with one input dataframe."
            )

        for d in [df, df2]:
            if d is not None:
                # assert that df has grouping_cols
                if not all(col in df.columns for col in self.grouping_cols):
                    PipelineCompileError(
                        f"Input dataframe must have all the grouping columns. Got : {df.columns}"
                    )
        # assert that self.layers is not empty
        if not self.layers:
            raise PipelineCompileError(
                msg="Pipeline must have at least one layer."
            )
        if not isinstance(self.output_schema, DataFrameSchema):
            raise PipelineCompileError(
                msg="Output schema cannot be mutiple. pyspark's applyInPandas only supports single output."
            )

    @property
    def input_schema(self) -> DataFrameSchema:
        """Input schema of the pipeline."""
        if len(self.layers) == 0:
            return None
        return self.layers[0].input_schema

    @property
    def output_schema(self) -> DataFrameSchema:
        """Output schema of the pipeline."""
        if len(self.layers) == 0:
            return None
        return self.layers[-1].output_schema

    def exception_handler(
        self, e: Exception, df: PandasDataFrame
    ) -> PandasDataFrame:
        """Exception handler of the pipeline.

        Args:
            e (Exception): Exception raised by the pipeline.
            df (DataFrame): Input DataFrame that caused the exception.

        Returns:
            DataFrame: DataFrame with the exception handler applied. The DataFrame must have the same schema as the output schema of the pipeline.

        Raises:
            Exception: If the exception handler does not return a DataFrame with the same schema as the output schema of the pipeline or
                if the exception handler is not passed and the pipeline raises an exception.
        """
        if self._exception_handler is not None:
            # call the exception handler
            df_exc = self._exception_handler(e, df)
            # validate the output of the exception handler
            df_exc = parse_dataframe_using_pandera_model(
                self.output_schema, df_exc
            )
            return df_exc
        else:
            raise e

    def _apply_layers(
        self, df: Union[PandasDataFrame, List[PandasDataFrame]]
    ) -> PandasDataFrame:
        """Apply all layers to the data to a single group."""
        _df = df.copy()
        for layer in self.layers:
            _df = layer.transform(_df)
        return _df

    def fit(
        self, df: Union[PandasDataFrame, List[PandasDataFrame]]
    ) -> PandasDataFrame:
        """Fit the pipeline to the data. Apply all layers to the data to a single group with an optional timeout."""
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._apply_layers, df)
                result = future.result(
                    timeout=self.timeout
                )  # Set the timeout here (in seconds)
            return result
        except Exception as e:
            return self.exception_handler(e, df)

    @typechecked
    def apply_in_pandas(self, df: PandasDataFrame) -> PandasDataFrame:
        """Apply the pipeline to the data using pandas 'groupby'. Apply all layers to the data grouped by the grouping columns.

        Args:
            df (DataFrame): DataFrame to apply the pipeline to.

        Returns:
            DataFrame: DataFrame with the pipeline applied.

        Raises:
            TypeCheckError: If df is not a DataFrame.
            PipelineCompileError: If the pipeline could not be compiled.
        """
        self.compile(df)
        grouped_df = df.groupby(self.grouping_cols)
        return grouped_df.apply(self.fit)

    @typechecked
    def apply_in_spark(
        self, df: SparkDataFrame, df2: Optional[SparkDataFrame] = None
    ) -> SparkDataFrame:
        """Apply the pipeline to the data using Spark 'groupBy'. Apply all layers to the data grouped by the grouping columns.
        In the case of multiple Spark DataFrames, the pipeline is applied in a cogroup fashion.

        Args:
            df (SparkDataFrame): Spark DataFrame to apply the pipeline to, or a list of Spark DataFrames.

        Returns:
            SparkDataFrame: Spark DataFrame with the pipeline applied.

        Raises:
            TypeCheckError: If df is not a SparkDataFrame.
            PipelineCompileError: If the pipeline could not be compiled.
        """

        self.compile(df, df2)

        grouped_df = df.groupby(self.grouping_cols)
        if df2 is not None:
            grouped_df = grouped_df.cogroup(df2.groupby(self.grouping_cols))

            def to_apply(key, left, right):
                return self.fit([left, right])

        else:

            def to_apply(key, df):
                return self.fit(df)

        return grouped_df.applyInPandas(
            to_apply,
            schema=pandera_model_to_spark_structype(self.output_schema),
        )

    def __repr__(self) -> str:
        rep = "PySparkPipe Pipeline\n"
        for layer in self.layers:
            rep += f"Layer: {layer.name}\n"
            rep += f"Input Schema: {layer.input_schema}\n"
            rep += f"Output Schema: {layer.output_schema}\n"
            rep += f"Transform: {layer.transform}\n"
            rep += f"Validate Input: {layer.validate_input}\n"
            rep += f"Validate Output: {layer.validate_output}\n"
            rep += "\n"

        return rep
