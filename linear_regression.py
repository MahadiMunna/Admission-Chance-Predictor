import findspark
findspark.init()

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("AdmissionChancePrediction").getOrCreate()

data = spark.read.csv("Admission.csv", header=True, inferSchema=True)
data.show(5)

data=data.withColumnRenamed('LOR ','LOR').withColumnRenamed('Chance of Admit ','Chance of Admit')
colNames = data.columns
print("columns: ",colNames)

data = data.drop("Serial No.")
for col in data.columns:
    print(col.ljust(20), data.filter(data[col].isNull()).count())

######## Linear Regression Model #########
from pyspark import SparkFiles
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.linalg import Vectors

assembler = VectorAssembler(
    inputCols=["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research"],
    outputCol="features")

df1 = assembler.transform(data)
final_data = df1.select("features", "Chance of Admit")
train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)

print('\nTotal train data - ',train_data.count())
train_data.show(5)
print('Total test data - ',test_data.count())
test_data.show(5)

lr = LinearRegression(featuresCol="features", labelCol="Chance of Admit")
lr_model = lr.fit(train_data)
result = lr_model.evaluate(train_data)
accuracy = float(result.r2)
print("\nAccuracy of Linear regression model is {:.2f}".format(accuracy),"\n")

predictions = lr_model.transform(test_data)
predictions.show(42)

import pandas as pd
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

data = [(320, 110, 5, 4.0, 4.5, 9.0, 1)]

pandas_df = pd.DataFrame(data, columns=["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research"])
user_df = spark.createDataFrame(pandas_df)

final_df = assembler.transform(user_df)
prediction = lr_model.transform(final_df).select("prediction").collect()[0]

print(f"Prediction: {prediction.prediction}")

lr_model.save("linear_regression_model")