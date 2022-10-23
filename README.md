# MovieEDA
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
spark = SparkSession.builder.appName("İstanbul BTK YAz Okulu MovieData EDA").getOrCreate()
rawDF = spark.read.option("Header",True).option("inferSchema",True).csv('/FileStore/tables/book_ratings.csv')
rawDF.printSchema()
root
 |-- userId: integer (nullable = true)
 |-- movieId: integer (nullable = true)
 |-- rating: double (nullable = true)
 |-- timestamp: integer (nullable = true)

rawDF.show()
+------+-------+------+----------+
|userId|movieId|rating| timestamp|
+------+-------+------+----------+
|     1|     31|   2.5|1260759144|
|     1|   1029|   3.0|1260759179|
|     1|   1061|   3.0|1260759182|
|     1|   1129|   2.0|1260759185|
|     1|   1172|   4.0|1260759205|
|     1|   1263|   2.0|1260759151|
|     1|   1287|   2.0|1260759187|
|     1|   1293|   2.0|1260759148|
|     1|   1339|   3.5|1260759125|
|     1|   1343|   2.0|1260759131|
|     1|   1371|   2.5|1260759135|
|     1|   1405|   1.0|1260759203|
|     1|   1953|   4.0|1260759191|
|     1|   2105|   4.0|1260759139|
|     1|   2150|   3.0|1260759194|
|     1|   2193|   2.0|1260759198|
|     1|   2294|   2.0|1260759108|
|     1|   2455|   2.5|1260759113|
|     1|   2968|   1.0|1260759200|
|     1|   3671|   3.0|1260759117|
+------+-------+------+----------+
only showing top 20 rows

rawDF.count()
Out[7]: 100004
movieByuserDF = rawDF.groupBy("userId").count()
movieByuserDF.show()
+------+-----+
|userId|count|
+------+-----+
|   148|  132|
|   463|  483|
|   471|  216|
|   496|  126|
|   243|  307|
|   392|   25|
|   540|   20|
|   623|  103|
|    31|   69|
|   516|  149|
|    85|  107|
|   137|   80|
|   251|  119|
|   451|   52|
|   580|  922|
|    65|   27|
|   458|   76|
|    53|   46|
|   255|  145|
|   481|  436|
+------+-----+
only showing top 20 rows

movieBymovieDF = rawDF.groupBy("movieId").count()
movieBymovieDF.count()
Out[11]: 9066
movieByuserDF.orderBy("count",ascending=False).show(5)
+------+-----+
|userId|count|
+------+-----+
|   547| 2391|
|   564| 1868|
|   624| 1735|
|    15| 1700|
|    73| 1610|
+------+-----+
only showing top 5 rows

display(movieByuserDF.orderBy("count",ascending=False))
avgRatingDF = rawDF.filter("movieId=429 or movieId=431").groupBy("movieId").avg("rating").withColumnRenamed("avg(rating)","avgByRating")
avgRatingDF.withColumn("fixAvgRating",round(col('avgByRating'),2)).show()
+-------+------------------+------------+
|movieId|       avgByRating|fixAvgRating|
+-------+------------------+------------+
|    429|2.5555555555555554|        2.56|
|    431|          3.671875|        3.67|
+-------+------------------+------------+

rawDF.select(countDistinct("userId")).withColumnRenamed("count(DISTINCT userId)","countByUserId").show()
+-------------+
|countByUserId|
+-------------+
|          671|
+-------------+

from pyspark.ml.recommendation import ALS
als = ALS(userCol="userId",itemCol="movieId",ratingCol="rating",coldStartStrategy="drop",nonnegative=True)
trainDF,testDF = rawDF.randomSplit([0.7,0.3])
rawDF.count()
Out[23]: 100004
trainDF.count()
Out[24]: 70043
testDF.count()
Out[25]: 29961
model = als.fit(trainDF)
predictDF = model.transform(testDF)
predictDF.show()
+------+-------+------+----------+----------+
|userId|movieId|rating| timestamp|prediction|
+------+-------+------+----------+----------+
|   148|     32|   4.0|1059603935|  4.180746|
|   148|     52|   4.0|1059504972| 3.7857811|
|   148|     58|   4.0|1059504946| 4.3171186|
|   148|    145|   2.0|1059530841| 3.5684235|
|   148|    588|   4.0|1059604244| 3.9501915|
|   148|    589|   3.5|1059507607| 4.1541605|
|   148|    648|   4.0|1059604568| 3.4902298|
|   148|    904|   5.0|1059604392|  4.276264|
|   148|    950|   3.5|1059604425| 4.5776224|
|   148|   1028|   5.0|1059505000| 4.2091365|
|   148|   1097|   5.0|1059604303| 3.9426484|
|   148|   1175|   4.0|1059603920| 4.5340877|
|   148|   1208|   5.0|1059504953| 4.3638096|
|   148|   1249|   3.5|1059507495|  4.228193|
|   148|   1269|   5.0|1059604396| 4.2067714|
|   148|   1391|   2.5|1059604044|  3.015143|
|   148|   1394|   4.5|1059504950| 4.2584157|
|   148|   1544|   3.5|1059604070|  2.959061|
|   148|   1610|   5.0|1059507616| 3.9595215|
|   148|   1653|   4.0|1059603970|  4.043335|
+------+-------+------+----------+----------+
only showing top 20 rows

