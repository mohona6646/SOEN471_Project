from data_processing import *
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

def knn(trainingSet,trainingSetLabels):
    printf("Fitting kNN model...")
    model = KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree', n_jobs=-1)
    model.fit(trainingSet, trainingSetLabels)
    printf(f"Finished fitting model.")

    printf("Predicting test labels...")

    predictions = model.predict(testSet)
    printf("Finished predictions.")

    printf("Storing results...")
    result = pd.DataFrame({'predictions': predictions, 'real_labels': testSetLabels})
    result.to_csv('./data/knn_results.csv', index=False)

def printf(*arg, **kwarg):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")
    print(f'[{timestamp}]', *arg, **kwarg)


def cleanup_data_and_store_as_parquet():
    players = data_preparation(
        "./data/male_players1.csv",
        "./data/male_players2.csv",
        "./data/male_players3.csv",
        "./data/male_players4.csv",
        "./data/male_players5.csv",
        "./data/male_players6.csv",
    )
    trainingSet, testSet = players.randomSplit([0.67, 0.33], 24)
    trainingSet.write.parquet('./data/male_players_cleaned_training')
    testSet.write.parquet('./data/male_players_cleaned_test')

if __name__ == "__main__":
    #cleanup_data_and_store_as_parquet()
    printf("-------Using the entire of data set-------")
    printf("Loading dataset...")
    # Retrieve training set and test set 
    trainingSet = pd.read_parquet('./data/male_players_cleaned_training')
    testSet = pd.read_parquet('./data/male_players_cleaned_test')
    printf(f'Finished loading dataset. Training set: {trainingSet.shape[0]} rows, Test set: {testSet.shape[0]} rows')
    # Get training set labels
    trainingSetLabels = trainingSet.iloc[:, -1:].values.ravel()
    # Drop the column "label_positions" of training set 
    trainingSet = trainingSet.iloc[:, :-1]
    
    # Get training set labels
    testSetLabels = testSet.iloc[:, -1:].values.ravel()
    # Drop the column "label_positions" of training set 
    testSet = testSet.iloc[:, :-1]

    #knn(trainingSet, trainingSetLabels)
    #printf("KNN-Done!")

    printf(f"Fitting Random Forest model...")


    ## Random Forest Classifier
    rdf = RandomForestClassifier(max_depth=7, random_state=0)
    rdf.fit(trainingSet, trainingSetLabels)
    printf(f"Finished fitting model.")

    printf("Predicting test labels...")

    predictions = rdf.predict(testSet)
    printf("Finished predictions.")

    printf("Storing results...")
    result = pd.DataFrame({'predictions': predictions, 'real_labels': testSetLabels})
    result.to_csv('./data/rdf_results.csv', index=False)
