# SOEN 471 Project

## Introduction and Research Question

Watching the 2022 FIFA World Cup, which debuted this past November, we got to witness a showcase of the best football talent from around the world. As classifying football players into different positions is a crucial task in the sport and helps to identify the strengths and weaknesses of each player and team, we want to determine if it is **possible to predict a player’s position** (forward, midfielder, defender, or goalkeeper) **based on their different skill ratings**, such as passing, shooting, dribbling or physique. The objective of our project is therefore to train a data model to classify football players into different positions, based on both their skill and performance characteristics, utilizing supervised machine learning.

## Dataset and Features

Our dataset used will be the [FIFA 23 Complete Player Dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset), procured from Kaggle, which contains all player data for the Career Mode from FIFA 15 to FIFA 23. We will only be maintaining the [_male_players.csv_](https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset?select=male_players.csv) file. Each player will be represented by a unique set of attributes and statistics, and our model will only work to learn the relationship between these attributes and the player’s most fitted position. Hence, data points related to the same player will not affect the prediction of the label, nor artificially inflate the performance of our classifier.

The procured dataset initially contains 10,003,590 unique values of player URLs, and a total of 110 columns for each player. We will be truncating this number of columns significantly, removing unnecessary characteristics such as name, height, etc., and only be maintaining the following 22 features: player_positions, shooting, passing, dribbling, defending, crossing, finishing, heading_accuracy, short_passing, volleys, dribbling, free_kick_accuracy, long_passing, ball_control, acceleration, sprint_speed, shot_power, stamina, long_shots, interceptions, positioning, and penalties.

## Model and Algorithms

For our project, we will be working with algorithms that are part of a **supervised classification** class of models. Our models will be trained on a labeled dataset, where the target variable (the player’s position) will be known. Our selected algorithms will then use features, such as passing, shooting, or dribbling, to learn the relationships between the features and the target variable, and then use this information to make predictions on new, unseen players.

Our training models will be based on three algorithms: decision tree, random forest, and k-nearest neighbors. For decision tree, each internal node will represent a decision based on a feature, while each leaf node will represent a prediction of the player’s position. Random forest will consider the prediction of each tree as a vote for a certain class, where the class with the most votes will result in a single, final prediction for each player. Lastly, the KNN algorithm will look at the _k_ closest data points to an unclassified player, based on their skill ratings, and based on what the neighboring data points are labeled, will make a prediction to classify the position.


## Our Team
|       Names         |   Github Username | Student ID | 
|    -------------    |   -------------   | ------------- |
| Mohona Mazumdar     | [mohona6646](https://github.com/mohona6646)       | 40129421   |
| Jasmine Lebel       | [jasminelebel](https://github.com/jasminelebel)  | 40135464    |
| Thi Mai Anh Nguyen  | [maianh2611](https://github.com/maianh2611)  | 40208493   |
| Ahmad Hamdy         | [ahmadhany99](https://github.com/ahmadhany99)  | 40068060   |
