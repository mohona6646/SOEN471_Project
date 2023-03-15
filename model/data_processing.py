# Spark imports
from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import desc

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
    players = df.count()

    return players


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
