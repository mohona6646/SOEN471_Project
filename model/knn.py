from data_processing import *
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def predict(model, name, trainingSet, trainingSetLabels):
    printf(f'Fitting {name} model...')
    model.fit(trainingSet, trainingSetLabels)
    printf(f"Finished fitting model.")

    printf("Predicting test labels...")

    predictions = model.predict(testSet)
    printf("Finished predictions.")

    # printf("Storing results...")
    # result = pd.DataFrame({'predictions': predictions, 'real_labels': testSetLabels})
    # result.to_csv('./data/{name}_results.csv', index=False)
    printf("Evaluating predictions...")
    recall = recall_score(testSetLabels, predictions, average='micro')
    printf('Recall: {}'.format(recall))
    presion = precision_score(testSetLabels, predictions, average='micro')
    printf('Precision: {}'.format(presion))
    f1Score = f1_score(testSetLabels, predictions, average='micro')
    printf('F1 Score: {}'.format(f1Score))
    printf("Confusion Matrix\n\n")
    cm = confusion_matrix(testSetLabels, predictions)
    printf(cm)

def printf(*arg, **kwarg):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")
    print(f'[{timestamp}]', *arg, **kwarg)


def cleanup_data_and_store_as_parquet(players):
    trainingSet, testSet = players.randomSplit([0.67, 0.33], 24)
    trainingSet.write.parquet('./data/male_players_cleaned_training')
    testSet.write.parquet('./data/male_players_cleaned_test')

if __name__ == "__main__":
    players = data_preparation(
        "./data/male_players.csv"
    )
    #cleanup_data_and_store_as_parquet(players)
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
    # KNN
    model = KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree', n_jobs=-1)
    predict(model,"KNN",trainingSet, trainingSetLabels)
    # Random Forest
    model = RandomForestClassifier(max_depth=7, random_state=0)
    predict(model,"Random Forest",trainingSet, trainingSetLabels)
    
    ## Create decision tree here