from pyspark.ml.classification import RandomForestClassifier
import matplotlib.pyplot as plt
from pyspark.sql.functions import col
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.types import IntegerType


def randomforest(df):

    #Convert the categorical column to numerical column
    indexer = StringIndexer(inputCol="label_position", outputCol="label_position_index")
    df = indexer.fit(df).transform(df)

    #One Hot
    # encoder = OneHotEncoder(inputCol='label_position_index', outputCol='OHElabel_position').fit(indexed)
    # df = encoder.transform(indexed)
    # df = df.drop("label_position_index", "label_position")

    list_of_features = df.drop("label_position").drop("label_position_index").columns
    #features = features.select([col(column).cast("integer") for column in features.columns])
    # for col_name in list_of_features:
    #     df = df.withColumn(col_name, col(col_name).cast(IntegerType()))
    assembler = VectorAssembler(inputCols=list_of_features, outputCol="indexed_features")
    df = assembler.transform(df)

    trainingData, testData = df.randomSplit([0.67, 0.33],24)
    rf = RandomForestClassifier(labelCol="label_position_index", featuresCol="indexed_features", numTrees=100, maxDepth=10)
    model = rf.fit(trainingData)
    predictions = model.transform(testData)
    evaluate_predictions(predictions)

    evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)

    print("Random Forest Model accuracy = %g" % (accuracy))
    return predictions

def evaluate_predictions(predictions):
    evaluator = BinaryClassificationEvaluator()
    print('Test Area Under ROC', evaluator.evaluate(predictions))

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    print('F1 score:', evaluator.evaluate(predictions))

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g " % (1.0 - accuracy))
    print("Accuracy = %g " % accuracy)



