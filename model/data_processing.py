# Spark imports
from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import desc, split

import findspark

findspark.init()


# Initialize a spark session.
def init_spark():
    spark = (
        SparkSession.builder.appName("Python Spark SQL basic example")
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


def data_preparation(filename):
    spark = init_spark()

    df = spark.read.csv(filename, header=True)

    # Start by selecting all field-related features relevant to our model and removing unnecessary characteristics
    # such as player name, height, age, net worth, etc.
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
    players = (
        players.drop(
            "goalkeeping_diving",
            "goalkeeping_handling",
            "goalkeeping_kicking",
            "goalkeeping_positioning",
            "goalkeeping_reflexes",
            "goalkeeping_speed",
        )
        .filter(players.player_position != "GK")
    )

    return players.take(2)


print(data_preparation("./data/male_players.csv"))
