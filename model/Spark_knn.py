from data_processing import *
import numpy as np
from pyspark.sql.functions import array, col, monotonically_increasing_id, row_number
from pyspark.sql.window import Window
from pyspark.mllib.evaluation import MulticlassMetrics
from sklearn.metrics import classification_report
import pandas as pd

def euclidean_distance(v1, v2):
    return np.sum((v1 - v2) ** 2)


def knnModel(trainingSet, toPredictSet, k):
    featuresArray = array(
        "pace",
        "shooting",
        "passing",
        "dribbling",
        "defending",
        "attacking_crossing",
        "attacking_finishing",
        "attacking_heading_accuracy",
        "attacking_short_passing",
        "attacking_volleys",
        "skill_dribbling",
        "skill_fk_accuracy",
        "skill_long_passing",
        "skill_ball_control",
        "movement_acceleration",
        "movement_sprint_speed",
        "movement_agility",
        "movement_balance",
        "power_shot_power",
        "power_stamina",
        "power_long_shots",
        "mentality_interceptions",
        "mentality_positioning",
        "mentality_penalties",
        "defending_marking_awareness",
        "defending_standing_tackle",
        "defending_sliding_tackle",
    )
    trainingSetDf = trainingSet.select(
        featuresArray.alias("features_in_array_training"), "label_position"
    )
    toPredictSetDf = toPredictSet.select(
        monotonically_increasing_id().alias("id"),
        featuresArray.alias("features_in_array_predict"),
    )
    merged = toPredictSetDf.crossJoin(trainingSetDf)
    mergedRDD = merged.rdd
    mergedRDD = mergedRDD.map(
        lambda r: [
            r.id,
            euclidean_distance(
                np.array(r.features_in_array_predict).astype(float),
                np.array(r.features_in_array_training).astype(float),
            ).tolist(),
            r.label_position,
        ]
    )
    merged = mergedRDD.toDF(["id", "distance", "label_position"])
    windowMerged = Window.partitionBy("id").orderBy(col("distance").asc())
    merged = (
        merged.withColumn("row", row_number().over(windowMerged))
        .filter(col("row") <= k)
        .drop("row")
        .drop("distance")
    )
    merged = merged.groupBy(["id", "label_position"]).count()
    windowMerged = Window.partitionBy("id").orderBy(col("count").desc())
    merged = (
        merged.withColumn("row", row_number().over(windowMerged))
        .filter(col("row") == 1)
        .drop("row")
        .drop("count")
    )
    merged = merged.orderBy("id").drop("id")
    merged = merged.withColumnRenamed("label_position", "prediction")
    return merged


if __name__ == "__main__":
    players = sampled_data()

    trainingSet, testSet = players.randomSplit([0.67, 0.33], 24)
    unlabeledTestSet, testSetRealLabels = testSet.drop(
        "label_position"
    ), testSet.select("label_position").withColumnRenamed(
        "label_position", "real_label"
    )

    testSetPredictions = knnModel(trainingSet, unlabeledTestSet, k=3)
    # Concat testSetPredictions and testSetRealLabels together and display results
    result = (
        testSetPredictions.withColumn("row_id", monotonically_increasing_id())
        .join(
            testSetRealLabels.withColumn("row_id", monotonically_increasing_id()),
            ("row_id"),
        )
        .orderBy(col("row_id").asc())
        .drop("row_id")
    )
    resultRDD = result.rdd
    resultPD = result.toPandas()
    evaluationMetrics = MulticlassMetrics(result)
    cr = classification_report(resultPD['real_label'],
                                              resultPD['prediction'], output_dict=True)
    report_df = pd.DataFrame.from_dict(cr).transpose()

    print("Classification Report:")
    print(report_df)
    confusion_matrix = evaluationMetrics.confusionMatrix().toArray()
    print("Confusion Matrix")
    print(confusion_matrix)