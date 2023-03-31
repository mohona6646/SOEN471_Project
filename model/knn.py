from data_processing import *
import pandas as pd
import pathlib
import shutil
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)


def predict(model, name, trainingSet, testSet, trainingSetLabels, testSetLabels):
    printf(f"Fitting {name} model...")
    model.fit(trainingSet, trainingSetLabels)
    printf(f"Finished fitting model.")

    printf("Predicting test labels...")

    predictions = model.predict(testSet)
    printf("Finished predictions.")

    # printf("Storing results...")
    # result = pd.DataFrame({'predictions': predictions, 'real_labels': testSetLabels})
    # result.to_csv('./data/{name}_results.csv', index=False)
    printf("Evaluating predictions...\n")
    recall = recall_score(testSetLabels, predictions, average="micro")
    presion = precision_score(testSetLabels, predictions, average="micro")
    f1Score = f1_score(testSetLabels, predictions, average="micro")
    accuracy = accuracy_score(testSetLabels, predictions)
    data = [[recall, presion, f1Score, accuracy]]
    df = pd.DataFrame(data, columns=["Recall", "Precision", "F1 Score", "Accuracy"])
    print(df)
    printf("Confusion Matrix")
    cm = confusion_matrix(testSetLabels, predictions)
    print(f"\n{cm}")


def printf(*arg, **kwarg):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")
    print(f"[{timestamp}]", *arg, **kwarg)


def cleanup_data_and_store_as_parquet(players):
    trainingSet, testSet = players.randomSplit([0.67, 0.33], 24)
    male_players_cleaned_training = pathlib.Path("../data/male_players_cleaned_training")
    if male_players_cleaned_training.exists():
         shutil.rmtree("../data/male_players_cleaned_training",ignore_errors=True)
    male_players_cleaned_test = pathlib.Path("../data/male_players_cleaned_test")
    if male_players_cleaned_test.exists():
        shutil.rmtree("../data/male_players_cleaned_test",ignore_errors=True)
        
    trainingSet.write.parquet('../data/male_players_cleaned_training')
    testSet.write.parquet('../data/male_players_cleaned_test') 


if __name__ == "__main__":
    players = data_preparation(
        "../data/male_players1.csv",
        "../data/male_players2.csv",
        "../data/male_players3.csv",
        "../data/male_players4.csv",
        "../data/male_players5.csv",
        "../data/male_players6.csv",
    )
    # cleanup_data_and_store_as_parquet(players)
    printf("-------Using the entire of data set-------")
    printf("Loading dataset...")
    # Retrieve training set and test set
    trainingSet = pd.read_parquet('../data/male_players_cleaned_training')
    testSet = pd.read_parquet('../data/male_players_cleaned_test')
    printf(
        f"Finished loading dataset. Training set: {trainingSet.shape[0]} rows, Test set: {testSet.shape[0]} rows"
    )
    # Get training set labels
    trainingSetLabels = trainingSet.iloc[:, -1:].values.ravel()
    # Drop the column "label_positions" of training set
    trainingSet = trainingSet.iloc[:, :-1]

    # Get training set labels
    testSetLabels = testSet.iloc[:, -1:].values.ravel()
    # Drop the column "label_positions" of training set
    testSet = testSet.iloc[:, :-1]
    # KNN
    model = KNeighborsClassifier(n_neighbors=3, algorithm="kd_tree", n_jobs=-1)
    predict(model, "KNN", trainingSet, trainingSetLabels, testSetLabels)
    # Random Forest
    model = RandomForestClassifier(max_depth=7, random_state=0)
    predict(model, "Random Forest", trainingSet, trainingSetLabels, testSetLabels)

    ## Create decision tree here
