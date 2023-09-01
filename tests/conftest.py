# -*- coding: utf-8 -*-
"""Conftest models."""


import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="module")
def spark():
    """Fixture to set up the in-memory spark with test data"""

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("pysparkpipe-test")
        .getOrCreate()
    )
    yield spark
