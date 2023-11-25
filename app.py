from flask import Flask, redirect, url_for, render_template, request

app = Flask(__name__)

@app.route("/",methods=['POST','GET'])
def submit():
    if request.method == "POST":
        gre = float(request.form["gre"])
        toefl = float(request.form["toefl"])
        uni_rate = float(request.form["uni_rate"])
        sop = float(request.form["sop"])
        lor = float(request.form["lor"])
        cgpa = float(request.form["cgpa"])
        research = float(request.form["research"])

        import findspark
        findspark.init()

        from pyspark.sql import SparkSession
        spark = SparkSession.builder.appName("AdmissionChancePrediction").getOrCreate()

        data = spark.read.csv("Admission.csv", header=True, inferSchema=True)

        data=data.withColumnRenamed('LOR ','LOR').withColumnRenamed('Chance of Admit ','Chance of Admit')

        data = data.drop("Serial No.")

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

        lr = LinearRegression(featuresCol="features", labelCol="Chance of Admit")
        lr_model = lr.fit(train_data)

        import pandas as pd
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

        input = [(gre, toefl, uni_rate, sop, lor, cgpa, research)]
        input_df = pd.DataFrame(input, columns=["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research"])
        input_df = spark.createDataFrame(input_df)

        input_df = assembler.transform(input_df)
        prediction = lr_model.transform(input_df).select("prediction").collect()[0]
        chance = prediction.prediction
        chance = int(chance*100)

        print(f"Prediction: {chance}")

        return redirect(url_for("user", prediction=chance))
    else:
        return render_template("index.html")

@app.route("/<prediction>")
def user(prediction):
    return render_template("prediction_page.html",content=prediction)


if __name__ == "__main__":
    app.run(debug=True)