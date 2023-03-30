# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Fundamentals of Pyspark
# MAGIC 
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC 
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.
# MAGIC in this notebook you will get fully idea of prectically implementation of pyspark.

# COMMAND ----------


# File location and type
file_location = "/FileStore/tables/insurance-1.csv"
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

df.printSchema()
# to get the information about dataset, as like .info()

# COMMAND ----------

df.show()
# to display dataset

# COMMAND ----------

df.show(5)
# to display the 5 first row

# COMMAND ----------

display(df)
# this function provided by databricks community. to display the dataset in a pretty format

# COMMAND ----------

df.head(10)
# this also work but it does not display the data in pretty format

# COMMAND ----------

spark.createDataFrame([("name",24,135,21),
                      ("class",23,162,14),
                      ("dob_year",1997,1998,99)],
                     ["Name","age","height","roll"]).show()

# to create a spark dataframe

# COMMAND ----------

spark_df = spark.createDataFrame([("name",24,135,21),
                      ("class",23,162,14),
                      ("dob_year",1997,1998,99)],
                     ["Name","age","height","roll"])

# COMMAND ----------

type(spark_df)

# COMMAND ----------

spark_df.show(5)

# COMMAND ----------

df.describe().show()

# to check the statiscal information about the dataset
# same like to the pandas

# COMMAND ----------

df.columns
# to check the name of all columns

# COMMAND ----------

df.dtypes
# to checkout the dtypes, same to the pandas

# COMMAND ----------

df.head()

# COMMAND ----------

type(df)

# COMMAND ----------

df.select('sex')
# returning selectec object of the column

# COMMAND ----------

df.select('sex').show(5)
# to select a single columns

# COMMAND ----------

sex = df.select('sex')
# assign a selected column into other variable

# COMMAND ----------

sex.show(5)

# COMMAND ----------

display(sex)

# COMMAND ----------

df.select(['sex','smoker']).show(5)
# to select multiple columns

# COMMAND ----------

df2 = df.select(['sex','smoker'])

# COMMAND ----------

df2.show(5)

# COMMAND ----------

type(df2)
## to check the dtypes selected two columns

# COMMAND ----------

type(sex)
# type of single columns, there is matter of series just like the pandas

# COMMAND ----------

df.collect()
# to display entire data in a single execution.

# COMMAND ----------

df.show(5)

# COMMAND ----------

type(df.dtypes)
# type output type

# COMMAND ----------

dict(df.dtypes)
# converted into dictionary

# COMMAND ----------

lst = []
for key,value in dict(df.dtypes).items():
    if value != 'string':
        lst.append(key)       
df.select(lst).show(5)

# to select the columns based on the dtypes
# all are numeric columns

# COMMAND ----------

lst = []
for key,value in dict(df.dtypes).items():
    if value == 'string':
        lst.append(key)       
df.select(lst).show(5)

# to select the columns based on the dtypes
# all are string columns

# COMMAND ----------

df.show(5)

# COMMAND ----------

df.withColumn('extra_charges',df['charges']+100).show(5)

# to add a new column, by using existing column

# COMMAND ----------

print(type(df.withColumn('extra_charges',df['charges']+100)))
df.withColumn('extra_charges',df['charges']+100).select('extra_charges').show(5)

# this was a spark dataframe that,s why , i directly applied the select function

# COMMAND ----------

to_drop_df = df.withColumn('extra_charges',df['charges']+100)
to_drop_df.show(5)

# now i will drop the 'extra_charges' columns

# COMMAND ----------

to_drop_df.drop('extra_charges').show(5)
# succesffully droped the columns

# COMMAND ----------

to_drop_df.drop(to_drop_df['extra_charges']).show(5)
# other types to define the column name inside the drop function

# COMMAND ----------

to_drop_df.drop('extra_charges','charges').show(5)
# to delete multiple columns directly pass the multiple columns no need to define list
# if you will define the list it will give you error

# COMMAND ----------

df.tail(5)
# it also work

# COMMAND ----------

df.withColumnRenamed('bmi','BMI').show(5)
# to renamed the column name

# COMMAND ----------

display(df.withColumnRenamed('bmi','BMI'))

#############   DATA VISUALIZATION BY THE DISPLAY FUNCTION            ###################

# COMMAND ----------

display(df)

#### profile the data, to take the quick revision of the data

# COMMAND ----------

df.show()

# COMMAND ----------

df.na.drop().show(5)
## to drop the null value
# how = can have any , all
# any <- if row contain any null then drop
# all <- only drop when a row, if row has all elements are null
# thresh = 2  <- if a row has 2 element is not null then row will not deleted
# subset = ['col_name']  <- if any null value in this 
# specific feature, that entire row will be deleted


# COMMAND ----------

df.na.fill('na').show(5)
# to fill the null value by 'na'

# COMMAND ----------

df.na.fill('na','age')
#fill by na only in this col_name

# COMMAND ----------

df.na.fill('na',['age','region'])
# you can specify multiple column names


# COMMAND ----------

## object initialization of imputer to impute na value to the mean
from pyspark.ml.feature import Imputer
imputer = Imputer(
    inputCols = ['age','bmi','children','charges'],
    outputCols =["{}_imputed".format(c) for c in ['age','bmi','children','charges']]).setStrategy("mean")

# to impute by mean , median ,

# COMMAND ----------

imputed_df = imputer.fit(df).transform(df)

# imputed the mean value and created new other columns, you will have to assign this value in other columns. bcz it dos,nt
# apply implicitly

# COMMAND ----------

imputed_df.show(5)

# COMMAND ----------

imputed_df.columns

# COMMAND ----------

df.columns
#

# COMMAND ----------

## filter operation in pyspark

df.filter('age<=18').show(5)

# COMMAND ----------

df.filter('age<=18').select(['age','smoker']).show()
# selected columns with filter operation

# COMMAND ----------

df.filter('age<=18').select(['age','smoker']).count()

# to count no of records

# COMMAND ----------

df.count(),len(df.columns)

# total no. of record and columns

# COMMAND ----------

df.filter(df['children']<=2).show(5)
# specify a column in a different way

# COMMAND ----------

##### FILTER operation with logic gate  ##############

# '&' for 'AND', '|' for 'OR', '~' for NOT

# COMMAND ----------

df.filter((df['children']<=2) & (df['smoker']=='yes')).show(5)
# filter with AND operation

# COMMAND ----------

df.filter((df['children']<=2) | (df['smoker']=='yes')).show(5)
# filter with OR operation

# COMMAND ----------

df.filter(df['sex']!='male').show(5)

# NOT equal too

# COMMAND ----------

df.filter(df['sex']!='male').count()
# female are 662

# COMMAND ----------

df.filter(df['sex']=='male').count()
# no. of all male

# COMMAND ----------

df.filter(~(df['sex']=='male')).count()

## NOT operator or INVERSE operator

# COMMAND ----------

########## groupby function   ###########

# COMMAND ----------

df.groupBy('smoker').count().show()

# COMMAND ----------

df.groupBy('children').count().show()

# COMMAND ----------

df.groupBy('sex').count().show()

# COMMAND ----------

df.groupBy('region').count().show()

# COMMAND ----------

df.groupBy('sex').count().show()

# COMMAND ----------

df.groupBy('sex').mean().show()
## mean of all numerical features with respect to sex

# COMMAND ----------

d = df.groupby('children').max()

# COMMAND ----------

d.select(['max(children)','max(charges)']).show(5)
## maximum charges on children

# COMMAND ----------

d.withColumnRenamed('max(charges)','expenses').show()

# COMMAND ----------

df.groupBy('children').sum().select(['sum(charges)','sum(bmi)']).show()
# groupby with select

# COMMAND ----------

df.agg({'age':'min','children':'min','charges':'sum'}).show()

### agg function to perform multiple aggregate function at a time

# COMMAND ----------

df.groupBy

# COMMAND ----------

df.groupBy('children').agg({'bmi':'mean'}).show()

# If your BMI is less than 18.5, it falls within the underweight range. 
# If your BMI is 18.5 to 24.9, it falls within the Healthy Weight 
# range. If your BMI is 25.0 to 29.9, it falls within the overweight range.
#If your BMI is 30.0 or higher, it falls within the obese range.

print('all are in overweight range.')

# COMMAND ----------

healthy_child = df.filter((df['bmi']>18.5)&(df['bmi']<24)).count()
print(f"only {healthy_child} are in healthy weight range ,in this dataset")
# If your BMI is 18.5 to 24.9, it falls within the Healthy Weight range.

# COMMAND ----------

############ spark mllib    ###############
with respect to spark mllib there are two techniques with respect to this.
1. RDD Api.
2. DataFrame Api.   <- this is quiet famous.

############    solve simple linear regression problem ##############


# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import Imputer
from pyspark.ml.regression import LinearRegression

# COMMAND ----------

## it combines your specify inputCols into independent feature
feature_assembler = VectorAssembler(inputCols = ['age','bmi','children'],outputCol = 'independent')

# COMMAND ----------

feature_assembler.transform(df).show(5)

# COMMAND ----------

data = feature_assembler.transform(df).select('independent','charges')

# COMMAND ----------

data.show(5)
# As you can see, data  is a combination 3 features and one dependent feature

# COMMAND ----------

data.printSchema()

# COMMAND ----------

data.count(),len(data.columns)
## total number of input

# COMMAND ----------

train_data,test_data = data.randomSplit([0.75,0.25])

# COMMAND ----------

train_data.count(),test_data.count()

# COMMAND ----------

# DBTITLE 1,display train_data
train_data.show(5)

# it has both X and Y

# COMMAND ----------

regression = LinearRegression(featuresCol='independent',labelCol='charges')

# set the value for independent and dependent variable

# COMMAND ----------

regresor_model = regression.fit(train_data)

# fit the algorithm on training data, it has both independent
# and dependent feauteres.

# COMMAND ----------

regresor_model.coefficients

# There were three features, and now three coefficients

# COMMAND ----------

regresor_model.intercept

# this is my intercept with respect to the LinearRegression

# COMMAND ----------

pred_result = regresor_model.evaluate(test_data)

# COMMAND ----------

pred_result.predictions.show(15)
# pyspark dataframe with X and Y and Y_pred

# COMMAND ----------

pred_result.meanAbsoluteError,pred_result.meanSquaredError

# error rate of my model

# COMMAND ----------

# MAGIC %md
# MAGIC Thank you 🤍🤍🤍
