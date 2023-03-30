# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Regression Problem  Solved by Pyspark Mllib
# MAGIC 
# MAGIC in this notebook, i will solve a regression problem with the help of PySpark.
# MAGIC all are steps  written in well commented.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import RandomForestRegressor

# from mllib library
from pyspark.mllib.tree import RandomForest, RandomForestModel
import numpy as np
import pandas as pd
print('libraries imported')

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/insurance-2.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

# Create a view or table

temp_table_name = "insurance"

df.createOrReplaceTempView(temp_table_name)
## if you want to apply the sql command on top of this dataset,
## you can create view of sql table

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC 
# MAGIC select age from insurance;

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.count(),len(df.columns)
# columns & row count

# COMMAND ----------

df.describe().show()

# COMMAND ----------

df.show(5)

# COMMAND ----------

df2 = df.na.drop()
df2.count()
#  null value removed

# COMMAND ----------

encoder = StringIndexer(inputCols=['sex','smoker','region'],outputCols=['sex_imputed','smokder_imputed','region_imputed'])
df3 = encoder.fit(df2).transform(df2)
df3.show(5)

# to impute the categorical value

# COMMAND ----------

assembler  =VectorAssembler(inputCols=['age','bmi','children','sex_imputed','smokder_imputed','region_imputed'],outputCol='Independent_features')
df4 = assembler.transform(df3)
df4.show(5)

# COMMAND ----------

final_data = df4.select(['Independent_features','charges'])
final_data.show(5)
# get the independent vector and dependent vector

# COMMAND ----------

train_data,test_data = final_data.randomSplit([0.75,0.25])
## train test split operation

# COMMAND ----------

train_data.count(),len(train_data.columns)

# COMMAND ----------

test_data.count(),len(test_data.columns)

# COMMAND ----------

train_data.show(5)

# COMMAND ----------

test_data.show(5)

# COMMAND ----------

random_for = RandomForestRegressor(featuresCol='Independent_features',labelCol='charges',maxDepth=9,)

# initialized object of RandomFOrestRegression class in pyspark

# COMMAND ----------

### tainin the model in pyspark

model = random_for.fit(train_data)

# COMMAND ----------

### get the prediction on test data
pred = model.transform(test_data)


# COMMAND ----------

pred.show(10)

# independent var  ,  dependent var   ,   prediction

# COMMAND ----------

display(pred)

### regression assumption by evaluate the model

# COMMAND ----------

residual = pred.withColumn('residual',pred['charges'] - pred['prediction'])
residual.show(5)

# COMMAND ----------

display(residual)

# COMMAND ----------

display(residual)

# COMMAND ----------

# MAGIC %md
# MAGIC Thank you 🤍🤍🤍🤍
