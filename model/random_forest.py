from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from sklearn.metrics import classification_report
import pandas as pd

def randomforest(df):

    #Convert the categorical column label_position to numerical column
    indexer = StringIndexer(inputCol="label_position", outputCol="label_position_index")
    df = indexer.fit(df).transform(df)

    #Drop the label_position column to only get the features
    list_of_features = df.drop("label_position").drop("label_position_index").columns

    assembler = VectorAssembler(inputCols=list_of_features, outputCol="indexed_features")
    df = assembler.transform(df)

    #Train random forest model and get the predictions from test set
    trainingData, testData = df.randomSplit([0.67, 0.33],24)
    rf = RandomForestClassifier(labelCol="label_position_index", featuresCol="indexed_features", numTrees=10, maxDepth=5)
    model = rf.fit(trainingData)
    predictions = model.transform(testData)

    #Generate classification ceport and confusion matrix
    predictions_and_labels = predictions.select("prediction", "label_position_index").rdd
    predictions_and_labels_pd= predictions.select("prediction", "label_position_index").toPandas()

    # Generate confusion matrix
    metrics = MulticlassMetrics(predictions_and_labels)
    confusion_matrix = metrics.confusionMatrix().toArray()

    # Generate classification report
    class_report_dict = classification_report(predictions_and_labels_pd['label_position_index'],
                                              predictions_and_labels_pd['prediction'], output_dict=True)
    class_report_df = pd.DataFrame.from_dict(class_report_dict).transpose()

    print("Classification Report:")
    print(class_report_df)
    print("\n")
    print("Confusion Matrix:")
    print(confusion_matrix)

    return predictions




