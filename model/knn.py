from data_processing import *
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
import os.path


def printf(*arg, **kwarg):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")
    print(f'[{timestamp}]', *arg, **kwarg)


def cleanup_data_and_store_as_parquet():
    players = data_preparation('./data/male_players.csv')
    trainingSet, testSet = players.randomSplit([0.67, 0.33], 24)
    trainingSet.write.parquet('./data/male_players_cleaned_training')
    testSet.write.parquet('./data/male_players_cleaned_test')

if __name__ == "__main__":
    # cleanup_data_and_store_as_parquet()

    printf("Loading dataset...")
    trainingSet = pd.read_parquet('./data/male_players_cleaned_training')
    testSet = pd.read_parquet('./data/male_players_cleaned_test')
    printf(f'Finished loading dataset. Training set: {trainingSet.shape[0]} rows, Test set: {testSet.shape[0]} rows')
    
    trainingSetLabels = trainingSet.iloc[:, -1:].values.ravel()
    trainingSet = trainingSet.iloc[:, :-1]

    testSetLabels = testSet.iloc[:, -1:].values.ravel()
    testSet = testSet.iloc[:, :-1]

    printf("Fitting kNN model...")
    model = KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree', n_jobs=-1)
    model.fit(trainingSet, trainingSetLabels)
    printf(f"Finished fitting model.")

    printf("Predicting test labels...")
    # chunks = 10000
    # testSetChunkSize = testSet.shape[0] // chunks
    # iteration = 0
    # for i in range(0, testSet.shape[0], testSetChunkSize):
    #     iteration += 1
    #     file_name = f'./data/knn_results/{iteration}_{chunks}.csv'
    #     if os.path.exists(file_name):
    #         print(f'Skipping iteration {iteration}')
    #         continue
    #     predictions = model.predict(testSet[i:i+testSetChunkSize])
    #     result = pd.DataFrame({'predictions': predictions})
    #     result.to_csv(file_name, index=False)
    #     printf(f'{iteration/(chunks/100)}% done')
    predictions = model.predict(testSet)
    printf("Finished predictions.")

    printf("Storing results...")
    result = pd.DataFrame({'predictions': predictions, 'real_labels': testSetLabels})
    result.to_csv('./data/knn_results.csv', index=False)

    printf("Done!")
