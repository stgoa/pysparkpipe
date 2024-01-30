# -*- coding: utf-8 -*-
"""Utils for interfacing between pandas, pandera.SchemaModel and pyspark."""


from typing import Dict, List, Union

import pandas as pd
import pyspark
from pandera import DataFrameSchema
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    BooleanType,
    DataType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

"""mapping from data type names to pyspark field types"""
TRANSFORM_DICT = {
    "int32": IntegerType(),
    "int64": LongType(),
    "Int64": LongType(),
    "float64": DoubleType(),
    "str": StringType(),
    "datetime64[ns]": TimestampType(),
    "bool": BooleanType(),
    "datetime64[ns, UTC]": TimestampType(),
    "datetime.date": DataType,
}


def create_spark_dataframe_using_pandera_model(
    cls: DataFrameSchema,
    spark: SparkSession,
    df: Union[pd.DataFrame, Dict, List[Dict]],
) -> pyspark.sql.DataFrame:
    """It converts the input into a spark dataframe following the class schema. Accept dictionaries, lists or dataframes as input.

    Args:
        cls (DataFrameSchema): Pandera class schema
        spark (SparkSession): SparkSession
        df (Union[pd.DataFrame,Dict, List[Dict]]): Input data

    Returns:
        pyspark.sql.DataFrame: Spark dataframe with the class schema
    """
    # parse and validate the input
    if isinstance(df, dict) or isinstance(df, list):
        df = parse_dataframe_from_dict(cls, df)
    else:
        df = parse_dataframe_using_pandera_model(cls, df)
    return spark.createDataFrame(
        df, schema=pandera_model_to_spark_structype(cls)
    )


def parse_dataframe_from_dict(
    cls: DataFrameSchema, data: Union[List[Dict], Dict], **kwargs
) -> pd.DataFrame:
    """
    Initializes dataframe from dict. Wrapper for pd.DataFrame() that adds the
    dtype=object argument to avoid unasked type-castings.

    After creating the dataframe, it is parsed (with the parse method), according
    to the schema.

    Args:
        cls (DataFrameSchema): Pandera class schema
        data (Union[List[Dict], Dict]): Dict, List of Dicts, or actually whatever
        input pd.DataFrame wouldn't sneeze at. In extra kwargs are required for
        compatibility, bring on the extra arguments, not gonna cry about that.

    Returns:
        pd.DataFrame: DataFrame post parsing.
    """

    df = pd.DataFrame(data=data, dtype=object, **kwargs)
    return parse_dataframe_using_pandera_model(cls, df)


def parse_dataframe_using_pandera_model(
    cls: DataFrameSchema, df: pd.DataFrame
) -> pd.DataFrame:
    """
    Parses DataFrame, adding required columns and doing valid type conversions
    according to schema typing. Includes a validation

    Args:
        cls (DataFrameSchema): Pandera class schema
        df (pd.DataFrame): pandas DataFrame

    Returns:
        pd.DataFrame: pandas DataFrame with correct types according to schema.
    """
    # Add missing columns to dataframe as empty columns
    df_cols = df.columns
    for column in cls.columns.values():
        if column.name not in df_cols and not column.regex:
            df[column.name] = pd.Series(dtype=str(column.dtype))

    # Validate types and convert
    df = cls.validate(df)
    return df


def pandera_model_to_spark_structype(
    cls: DataFrameSchema, df_cols: List[str] = None
) -> StructType:
    """
    Returns spark StructType with schema or corresponding Pandera schema.

    Args:
        cls (DataFrameSchema): Pandera class schema
        df_cols (Optional[List[str]], optional): List of DataFrame col names of corresponding schema,
        useful for returning StructType with a particular column order.
        Defaults to None. If None, uses default schema column ordering.

    Returns:
        StructType: schema for pyspark DataFrame
    """

    schema_cols = cls.columns

    if df_cols is None:
        df_cols = [col for col in schema_cols.keys()]

    struct = StructType()
    for col in df_cols:
        struct.add(
            StructField(
                col,
                TRANSFORM_DICT[str(schema_cols.get(col).dtype)],
                schema_cols.get(col).nullable,
            )
        )
    return struct
