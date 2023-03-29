# Spark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, when, col

import findspark
from pyspark.sql.types import IntegerType

import random_forest
import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

findspark.init()


# Initialize a spark session.
def init_spark():
    spark = (
        SparkSession.builder.appName("SOEN 471 Project")
        .config("spark.some.config.option", "some-value")
        .getOrCreate()
    )
    return spark


# Converts FIFA-defined positions to one of our three labels.
def label_conversion(player_position):
    player_position = (
        when(player_position.isin(["CB", "RB", "LB", "RWB", "LWB"]), "Defender")
        .when(player_position.isin(["CM", "CDM", "CAM", "RM", "LM"]), "Midfielder")
        .when(player_position.isin(["ST", "CF", "RF", "LF", "RW", "LW"]), "Forward")
        .otherwise("Undefined")
    )

    return player_position


# This data preparation phase returns a count of 3,266,866 defenders, 3,712,577 midfielders,
# and 1,902,995 forwards. Majority within the defender and midfielders classes is visible.
def data_preparation(file1):
    spark = init_spark()

    df1 = spark.read.csv(file1, header=True)
    # df2 = spark.read.csv(file2, header=True)
    # df3 = spark.read.csv(file3, header=True)
    # df4 = spark.read.csv(file4, header=True)
    # df5 = spark.read.csv(file5, header=True)
    # df6 = spark.read.csv(file6, header=True)

    df = df1 #.union(df2).union(df3).union(df4).union(df5).union(df6)

    # Start by selecting all field-related features relevant to our model and removing
    # unnecessary characteristics such as player name, height, age, net worth, etc.
    # Initial row count is of 10,003,590.
    players = df.select(
        "player_positions",
        "overall",
        "skill_moves",
        "pace",
        "shooting",
        "passing",
        "dribbling",
        "defending",
        "physic",
        "attacking_crossing",
        "attacking_finishing",
        "attacking_heading_accuracy",
        "attacking_short_passing",
        "attacking_volleys",
        "skill_dribbling",
        "skill_curve",
        "skill_fk_accuracy",
        "skill_long_passing",
        "skill_ball_control",
        "movement_acceleration",
        "movement_sprint_speed",
        "movement_agility",
        "movement_reactions",
        "movement_balance",
        "power_shot_power",
        "power_jumping",
        "power_stamina",
        "power_strength",
        "power_long_shots",
        "mentality_aggression",
        "mentality_interceptions",
        "mentality_positioning",
        "mentality_vision",
        "mentality_penalties",
        "mentality_composure",
        "defending_marking_awareness",
        "defending_standing_tackle",
        "defending_sliding_tackle",
        "goalkeeping_diving",
        "goalkeeping_handling",
        "goalkeeping_kicking",
        "goalkeeping_positioning",
        "goalkeeping_reflexes",
        "goalkeeping_speed",
    )

    # Most players have more than one preferred position, displayed in increasing order.
    # We therefore filter through the positions and return only the most preferred,
    # creating a new column and dropping the previous one.
    players = players.withColumn(
        "player_position", split(players["player_positions"], ",")[0]
    ).drop("player_positions")

    # For players whose position is goalkeeper, important attribute fields are empty,
    # such as shooting, passing, dribbling, defending, etc.
    # We will therefore remove all goal-keeping related attributes and goal-keepers
    # from our data selection.
    # Reduces our row count from 10,003,590 to 8,882,644.
    players = players.drop(
        "goalkeeping_diving",
        "goalkeeping_handling",
        "goalkeeping_kicking",
        "goalkeeping_positioning",
        "goalkeeping_reflexes",
        "goalkeeping_speed",
    ).filter(players.player_position != "GK")

    # Convert specific positions such as RW, CDM, LB to a more general
    # label (defender, midfielder or forward).
    players = players.withColumn(
        "label_position", label_conversion(players["player_position"])
    ).drop("player_position")

    # Drop features which are not discriminative towards any of the three labels,
    # and where same values can easily be obtained between players of all classes.
    players = players.drop(
        "overall",
        "skill_moves",
        "physic",
        "skill_curve",
        "movement_reactions",
        "power_jumping",
        "power_strength",
        "mentality_aggression",
        "mentality_vision",
        "mentality_composure",
    )

    # Drop rows containing null values or a label position of "Undefined"
    # No undefined positions were counted but removing nulls minimally reduces
    # our row count from 8,882,644 to 8,882,438.
    players.filter(players.label_position != "Undefined").count()
    players = players.dropna()

    # Defenders: 3,266,866, Midfielders: 3,712,577, Forwards: 1,902,995, Undefined: 0
    for col_name in players.columns:
        if col_name != "label_position":
            players = players.withColumn(col_name, col(col_name).cast(IntegerType()))
    return players


# Undersamples the data to reduce class imbalance, returning 1,500,000
# player values for each of the three classes (easily splittable into
# two thirds training and one third testing).
def sampled_data():
    players = data_preparation(
        "../data/male_players.csv"
        # "./data/male_players2.csv",
        # "./data/male_players3.csv",
        # "./data/male_players4.csv",
        # "./data/male_players5.csv",
        # "./data/male_players6.csv",
    )

    # Filter through classes by position name.
    defenders = players.filter(players.label_position == "Defender")
    midfielders = players.filter(players.label_position == "Midfielder")
    forwards = players.filter(players.label_position == "Forward")

    # Sample 1,500,000 player values from each class to remove data imbalance where
    # the Forwards class is a minority.
    defenders = defenders.sample(fraction=500 / defenders.count()).limit(500)
    midfielders = midfielders.sample(fraction=500 / midfielders.count()).limit(
        500
    )
    forwards = forwards.sample(fraction=500 / forwards.count()).limit(500)

    players = defenders.union(midfielders).union(forwards)

    return players


df = sampled_data()
random_forest.randomforest(df)

