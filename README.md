# PySpark Pipe

**PySparkPipe** is a Python library designed to simplify the development of data pipelines within the PySpark framework. It provides a powerful and intuitive way to apply a sequence of transformations over grouped data within the context of [pyspark.sql.GroupedData.applyInPandas](https://spark.apache.org/docs/3.2.1/api/python/reference/api/pyspark.sql.GroupedData.applyInPandas.html)

# Why PySparkPipe?
Here are some reasons why PySparkPipe is a valuable addition to your data engineering/science toolkit:

1. Grouped data transformation in a single reshuffle.
    In the realm of large-scale ETL pipelines, transitioning to PySpark for distributed data processing is crucial for scalability. However, complex transformations involving grouping often lead to performance bottlenecks due to data shuffling. This is where PySparkPipe becomes indispensable. It provides a structured and efficient way to define your custom transformations for each group while ensuring minimal data shuffling—avoiding the significant data shuffling that can hinder performance and scalability when transformations are applied individually within each group. You can model each transformation step using PySparkPipe's Layers, maintaining the modularity and clarity of your code. When it's time to execute the pipeline, PySparkPipe intelligently combines these steps into a single reshuffling operation, optimizing data flow across the cluster. This reduces the computational overhead and network traffic associated with multiple shuffles, resulting in significantly faster ETL processing.

2. Consistency and validation.
    The Layer class in PySparkPipe ensures that each intermediate transformation within your pipeline is validated for consistency. This means you can catch data issues early in the pipeline, reducing the risk of downstream errors and enhancing the quality of your data.

3. Integration with PySpark and Pandas.
    PySparkPipe gives you the flexibility to choose between PySpark and pandas for your data transformations. Whether you need the scalability of PySpark or the convenience of pandas, you can seamlessly switch between them within your pipeline.

4. Scalability.
    Leveraging the power of PySpark, PySparkPipe can handle large-scale data processing tasks, making it suitable for big data applications.

5. Community and contribution.
    y using PySparkPipe, you join a community of data engineers and developers who can contribute to the library's growth and improvement. We welcome contributions and value the collective knowledge and experience of the community.

6. Simplified maintenance.
    With a well-structured pipeline and clear separation of transformation layers, PySparkPipe makes pipeline maintenance and troubleshooting more straightforward. This reduces the long-term effort required to maintain your data processing workflows.


## Efficient Grouped Data Transformation in a Single Reshuffle

Performing Python transformations separately can often lead to performance bottlenecks, primarily due to the overhead of data shuffling.

<p align="center">
<img src="https://lucid.app/publicSegments/view/a46f39f2-0d78-469e-bcfc-7e668d9676a4/image.png" alt="pysparkpipe-sequential" style="width:1200px;"/>
</p>

However, with PySparkPipe, you ensure that each partition group is processed within the same core, eliminating unnecessary data shuffling and significantly improving overall efficiency.

<p align="center">
<img src="https://lucid.app/publicSegments/view/6ee4f386-a66b-4183-ad71-4b38d46c5227/image.png" alt="pysparkpipe-parallel" style="width:1200px;"/>
</p>

PySparkPipe streamlines your data processing workflows, allowing you to harness the full potential of distributed computing without sacrificing performance.



# Installation

This project uses:

-   Python >= 3.9
-   [Poetry](https://python-poetry.org/) Only needed for installation from git repository. Installing wheels from private PyPI doesn't require it.


## Cloning from git repository

```sh
    git clone https://github.com/stgoa/pysparkpipe.git
    cd pysparkpipe
    python -m venv .venv
    poetry install
```

## Installing from pip (soon)

For installation using ``pip`` run

```sh
pip install pysparkpipe
```

# Features

- **Pipeline Class:** The core of the library is the `Pipeline` class, which serves as a tool for orchestrating a sequence of transformations over grouped data. These transformations can be executed either using pandas or PySpark.

- **Layer Class:** Within the pipeline module, the `Layer` class plays a crucial role in managing intermediate transformations. It validates the data and applies specific transformations, ensuring the consistency and quality of your pipeline.

- **Accumulation and Orchestration:** The `Pipeline` class accumulates and orchestrates `Layers`. You can easily add new layers to the sequence using the "add" method. Each added layer is validated for consistency, ensuring a smooth pipeline flow.

- **Grouped Data Processing:** The "apply_in_spark" method in the `Pipeline` class is designed to take an input Spark DataFrame, group it, and then apply the sequence of layers to each grouped DataFrame. This streamlined process simplifies complex grouped data processing tasks.


# Usage

This project implements several time series preprocessing


```python

import numpy as np
import pandas as pd
from pandera import Column, DataFrameSchema
from pyspark.sql import SparkSession
from pysparkpipe.pipeline import Pipeline


spark = (
    SparkSession.builder.master("local[2]")
    .appName("pysparkpipe-test")
    .getOrCreate()
)

# input data
df = spark.createDataFrame(pd.DataFrame(data={"col1": [1, 1, "2", "2", "3", "3"], "col2": [0, 1, 0, 2, 0, 3]}))

df.show()
"""
+----+----+
|col1|col2|
+----+----+
|   1|   0|
|   1|   1|
|   2|   0|
|   2|   2|
|   3|   0|
|   3|   3|
+----+----+
"""

# create a pipeline
pipe = Pipeline(
    grouping_cols=["col1"],
    validate_inputs=True,
    validate_outputs=True,
    )

# create an input schema
input_schema = DataFrameSchema(
    {
        "col1": Column(str, nullable=False, coerce=True),
        "col2": Column(float, nullable=False, coerce=True),
    }
    )

# create a transformation function
def transform_max(x):
    col1 = [x["col1"].iloc[0]]
    col2 = [np.max(x["col2"])]
    return pd.DataFrame({"col1": col1, "col2": col2})


# create an output schema
output_schema = DataFrameSchema(
    {
        "col1": Column(int, nullable=False, coerce=True),
        "col2": Column(float, nullable=False, coerce=True),
    }
    )

# add a layer to the pipeline
pipe.add(transform_max, input_schema, output_schema)

# apply the pipeline to the input data
df_output = pipe.apply_in_spark(df)

df_output.show()
"""
+----+----+
|col1|col2|
+----+----+
|   1| 1.0|
|   2| 2.0|
|   3| 3.0|
+----+----+
"""

# add another layer to the pipeline
def multiply(x):
    x["col2"] = x["col2"] * 2.0
    return x

pipe.add(multiply, output_schema, output_schema)

# apply the pipeline to the input data
df_output = pipe.apply_in_spark(df)

df_output.show()
"""
+----+----+
|col1|col2|
+----+----+
|   1| 2.0|
|   2| 4.0|
|   3| 6.0|
+----+----+
"""

```

## Developing

To create new features and improve the project you must set a development enviroment, because you need to use Spark. There are two needed things to take in mind. First, you will need an enviroment where you can test your code in programming time, and a jupyter notebook is a very nice option. We recommend to use the [pyspark-notebook](https://hub.docker.com/r/jupyter/pyspark-notebook) and mount a volume of the code folder ``pysparkpipe`` inside a container of that image (here ``pysparkpipe`` is the inner folder where are placed all ``.py`` files). The volume will help to update all the pysparkpipe files inside the container at programming time.

### Testing

For testing, we don't need ``pyspark-notebook``, because that is a development tool. We use a minimal Docker image with a Spark environment in order to run the integration test, this image is build with the ``Dockerfile`` of the project. The test will run with the Azure Pipeline as part of the CI process. We also recommend to create a local container with a mounted volume in order to run the test local before going into the pipeline. To do that we do the following commands

```sh
docker build -t pysparkpipe_tester_image --target tester .
```
This command will create an image that allow us to create containers with ``pysparkpipe`` ``.py`` files and the ``tests`` folder. Next you need to create a container with a mounted volume to update the ``.py`` files at programming time

```sh
docker run -v <FULL_PATH_TO_PYSPARKPIPE_PY_FILES_FOLDER>:/pysparkpipe/pysparkpipe -v <FULL_PATH_TO_PYSPARKPIPE_PY_FILES_FOLDER>:/pysparkpipe/tests  --name pysparkpipe_test_container pysparkpipe_tester_image
```

The command above will create a container called ``pysparkpipe_test_container`` with a volume between our local ``.py`` files and the container.


## TODO

* DAG pipeline structures
*
