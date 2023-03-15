# Spark imports
from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, when

import findspark

findspark.init()


# Initialize a spark session.
def init_spark():
    spark = (
        SparkSession.builder.appName("SOEN 471 Project")
        .config("spark.some.config.option", "some-value")
        .getOrCreate()
    )
    return spark


# Useful functions to print RDDs and Dataframes.
def toCSVLineRDD(rdd):
    """
    This function convert an RDD or a DataFrame into a CSV string
    """
    a = rdd.map(lambda row: ",".join([str(elt) for elt in row])).reduce(
        lambda x, y: "\n".join([x, y])
    )
    return a + "\n"


def toCSVLine(data):
    """
    Convert an RDD or a DataFrame into a CSV string
    """
    if isinstance(data, RDD):
        return toCSVLineRDD(data)
    elif isinstance(data, DataFrame):
        return toCSVLineRDD(data.rdd)
    return None


# Converts FIFA-defined positions to one of our three labels.
def label_conversion(player_position):
    player_position = (
        when(player_position.isin(["CB", "RB", "LB", "RWB", "LWB"]), "Defender")
        .when(player_position.isin(["CM", "CDM", "CAM", "RM", "LM"]), "Midfielder")
        .when(player_position.isin(["ST", "CF", "RF", "LF", "RW", "LW"]), "Forward")
        .otherwise("Undefined")
    )

    return player_position


def data_preparation(file1, file2, file3, file4, file5, file6):
    spark = init_spark()

    df1 = spark.read.csv(file1, header=True)
    df2 = spark.read.csv(file2, header=True)
    df3 = spark.read.csv(file3, header=True)
    df4 = spark.read.csv(file4, header=True)
    df5 = spark.read.csv(file5, header=True)
    df6 = spark.read.csv(file6, header=True)

    df = df1.union(df2).union(df3).union(df4).union(df5).union(df6)

    # Start by selecting all field-related features relevant to our model and removing unnecessary characteristics
    # such as player name, height, age, net worth, etc.
    # Initial row count is of 10,003,590.
    players = df.select(
        "long_name",
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

    # For players whose position is not goalkeeper, most of these fields are empty.
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

    players = players.withColumn(
        "label_position", label_conversion(players["player_position"])
    ).drop("player_position")

    # Drop rows containing null values or a label position of "Undefined"
    # No undefined positions were counted but removing nulls reduces our
    # row count from ... to ...
    # players.filter(players.label_position != "Undefined").count()
    # players = players.dropna()

    defenders = players.filter(players.label_position == "Defender")
    midfielders = players.filter(players.label_position == "Midfielder")
    forwards = players.filter(players.label_position == "Forward")
    undefined = players.filter(players.label_position == "Undefined")

    # Defenders = 3,266,916, Midfielders: 3,712,669, Forwards: 1,903,059, Undefined: 0
    return (defenders.count(), midfielders.count(), forwards.count(), undefined.count())


print(
    data_preparation(
        "./data/male_players1.csv",
        "./data/male_players2.csv",
        "./data/male_players3.csv",
        "./data/male_players4.csv",
        "./data/male_players5.csv",
        "./data/male_players6.csv",
    )
)
