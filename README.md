# SOEN 471 Project

## Introduction and Research Question

Watching the 2022 FIFA World Cup, which debuted this past November, we got to witness a showcase of the best football talent from around the world. This tournament provided both insight and data on players’ performance and abilities, with players in each position playing to their strengths. As classifying football players into different positions is a crucial task in the sport, and helps to identify the strengths and weaknesses of each player and team, we decided to attempt predicting a player’s position based on their different skill ratings, such as passing, shooting, dribbling or physique. The objective of this project is therefore to train a data model to classify football players into different positions, based on both their skill and performance characteristics, utilizing supervised machine learning.

## Dataset and Features

In terms of dataset, we decided that the more obvious fit was the [FIFA 23 Complete Player Dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset), procured from Kaggle, which contains all player data for the Career Mode from FIFA 15 to FIFA 23. For this project, we discarded files containing coaches, teams, and female players, and only maintained the [_male_players.csv_](https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset?select=male_players.csv) file. As this data allows multiple comparisons for the same players across the last nine versions of the video game, one may inquire about data replication between the training and test sets. However, performance statistics for each player are updated every year, and we are not using any player names. Each player is represented by a unique set of attributes and statistics, and our model only works to learn the relationship between these attributes and the player’s most fitted position. Hence, these data points do not affect the prediction of the label, nor artificially inflate the performance of our classifier.

The procured dataset initially contains **10,003,590** unique values of player URLs, and a total of 110 columns for each player. These include personal characteristics (such as name or age), physical attributes (height, weight, body type, etc.), and other statistics such as technical, tactical, or mental skills. As we aim to provide the most accurate prediction concerning the relationship between player performance statistics and position, we therefore truncated this number of columns significantly. As a result, we only maintained the following **27 features**: pace, shooting, passing, dribbling, defending, attacking_crossing, attacking_finishing, attacking_heading_accuracy, attacking_short_passing, attacking_volleys, skill_dribbling, skill_fk_accuracy, skill_long_passing, skill_ball_control, movement_acceleration, movement_sprint_speed, movement_agility, movement_balance, power_shot_power, power_stamina, power_long_shots, meantality_interceptions, mentality_positioning, mentality_penalties, defending_marking_awareness, defending_standing_tackle, and defending_sliding_tackle.

A football player is classified among one of four categories: forward, midfielder, defender, or goalkeeper. However, as we have removed all goalie-specific features contained in the initial dataset, such as goalkeeping_diving, goalkeeping_kicking, etc., we decided to discard all goalkeepers from our dataset. These players contain no data for our selected features, such as passing, shooting or dribbling, so we therefore deemed completely retrieving them most appropriate. As a result, all data points are classified among one of three labels: **forward**, **midfielder**, or **defender**.

## Model and Algorithms

For our project, we are working with algorithms that are part of a **supervised classification** class of models. The goal of supervised learning is to build a model that can make predictions on new, unseen data based on the relationships learned from the training data. In our case, our models are trained on a labeled dataset, where the target variable (the player’s position) is known. Our selected algorithms then use features, such as passing, shooting, or dribbling, to learn the relationships between the features and the target variable, and then use this information to make predictions on new, unseen players.

Our training models are based on **three** algorithms: Decision Tree, Random Forest and K-Nearest Neighbors (KNN). 

The **Decision Tree** algorithm operates by recursively dividing the tree into various characteristics and choosing, at each node, the attribute that provides the most effective division of the data. In our case, each internal node represents a decision based on a feature (ex: passing, shooting, dribbling), while each leaf node represents a prediction of the player’s position. The algorithm then evaluates the information gain to determine the optimal sequence of attributes for splitting the tree. However, decision trees tend to overfit, which means they may pay too much attention to minor details in the training set, causing poor performance on the testing set. 

**Random Forest** is an algorithm that creates multiple decision trees using random subsets of the data and features and combines the results for a final prediction. The algorithm considers the prediction of each tree as a vote for a certain class and the class with the most votes is chosen to be the predicted class, resulting in a single, final prediction for each player. This algorithm works to reduce overfitting and increase the overall accuracy of the model. However, due to the requirement to build multiple decision trees and combine their predictions, the learning process for the Random Forest algorithm is generally slower than that of the KNN algorithm, particularly when dealing with large datasets. 

Lastly, the **K-Nearest Neighbors** algorithm classifies a new data point based on the characteristics of its neighbors. Therefore, KNN looks at the k-closest data points to an unclassified player, based on their skill ratings, and based on what the neighboring data points are labeled, makes a prediction to classify the position.  Although the KNN algorithm is simple and fast to learn, it is less tolerant of missing values, compared to Random Forest. In the case of missing values, the algorithm may struggle to accurately find the closest neighbors, which can affect the accuracy of its predictions. In contrast, Random Forest handles missing values better as it combines multiple decision tree predictions, which neutralizes the effect of missing values across its trees. 

Using these three algorithms and comparing the performance of each, we hope that our model can provide an objective and data-driven evaluation of player abilities and potential that can eventually be used to support decision-making in the football industry.

## Cloning the repository

Once the repository has been cloned, the split data files are still so large that you may find your `data` folder to be empty. In that case, the complete dataset can be downloaded from Kaggle ([here](https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset?select=male_players.csv)) and placed into your folder instead. You need only to update the file name the dataframe takes in as input in `data_processing.py` after that.

#### Required Libraries

- pandas
- matplotlib
- pathlib
- shutil
- numpy
- pyspark
- findspark
- sklearn

## Our Team
|       Names         |   Github Username | Student ID | 
|    -------------    |   -------------   | ------------- |
| Mohona Mazumdar     | [mohona6646](https://github.com/mohona6646)       | 40129421   |
| Jasmine Lebel       | [jasminelebel](https://github.com/jasminelebel)  | 40135464    |
| Thi Mai Anh Nguyen  | [maianh2611](https://github.com/maianh2611)  | 40208493   |
| Ahmad Hamdy         | [ahmadhany99](https://github.com/ahmadhany99)  | 40068060   |
