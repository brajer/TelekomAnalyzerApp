package distance

import io.{HaversineDistanced, HaversineKeyValue, ReducedTelekomEvent}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Dataset, Row, SparkSession}

final case class HaversineRoundedKeyValue(dataset: String, haversine: Int)

object DistanceAnalyzer {

  val dsPartitioner = Window.partitionBy("dataset")
  val orderedDsPartitioner = Window.partitionBy("dataset").orderBy("haversine")
  val reverseDsPartitioner = Window.partitionBy("dataset").orderBy(col("haversine").desc)



  def getDistanceMetaDataPerDay(spark: SparkSession, ds: Dataset[HaversineKeyValue]): Dataset[_] = {
    println("...Metadata [PER DAY]")

    ds
      .withColumn("count", count("haversine").over(dsPartitioner))
      .withColumn("min", min("haversine").over(dsPartitioner))
      .withColumn("max", max("haversine").over(dsPartitioner))
      .withColumn("mean", mean("haversine").over(dsPartitioner))
      .drop("haversine")
      .distinct()
      .orderBy("dataset")
  }

  def getDistanceQuantilesPerDay(spark: SparkSession, ds: Dataset[HaversineKeyValue]): Dataset[_] = {
    import spark.implicits._

    println("...Quantiles [PER DAY]")

    ds
      .withColumn("ntile", ntile(4).over(orderedDsPartitioner))
      .map{ case Row(dataset: String, haversine: Double, ntile: Int) =>
        ((dataset, ntile), haversine)
      }
      .groupByKey(_._1)
      .reduceGroups((dist1, dist2) => (dist1._1, math.min(dist1._2, dist2._2)))
      .map(keyValue => (keyValue._1._1, keyValue._1._2, keyValue._2._2))
      .withColumnRenamed("_1", "dataset")
      .withColumnRenamed("_2", "quantile")
      .withColumnRenamed("_3", "haversine")
      .orderBy("dataset", "quantile")
  }

  def getDistanceQuantilesAll(spark: SparkSession, ds: Dataset[HaversineKeyValue]): Dataset[_] = {
    import spark.implicits._

    println("...Quantiles [ALL]")

    ds.select("haversine")
      .withColumn("unified", lit(1))
      .map(haversine =>
        HaversineKeyValue(haversine.getInt(1).toString, haversine.getDouble(0))
      )
      .withColumn("ntile", ntile(4).over(orderedDsPartitioner))
      .map{ case Row(dataset: String, haversine: Double, ntile: Int) =>
        ((dataset, ntile), haversine)
      }
      .groupByKey(_._1)
      .reduceGroups((dist1, dist2) => (dist1._1, math.min(dist1._2, dist2._2)))
      .map(keyValue => (keyValue._1._1, keyValue._1._2, keyValue._2._2))
      .select("_2", "_3")
      .withColumnRenamed("_2", "quantile")
      .withColumnRenamed("_3", "haversine")
      .orderBy("quantile")
  }

  def getDistanceCumulativeDistributionPerDay(spark: SparkSession, ds:Dataset[HaversineKeyValue]): Dataset[_] = {
    import spark.implicits._

    println("...Distance Cumulative distribution [PER DAY]")

    ds
      .map{ haversineKeyValue: HaversineKeyValue =>
        HaversineRoundedKeyValue(haversineKeyValue.dataset, math.round(haversineKeyValue.haversine).toInt)
      }
      .withColumn("cume_dist", cume_dist().over(orderedDsPartitioner))
      .distinct()
      .orderBy("dataset", "haversine")
  }

  def getDistanceCumulativeDistributionAll(spark: SparkSession, ds:Dataset[HaversineKeyValue]): Dataset[_] = {
    import spark.implicits._

    println("...Distance Cumulative distribution [ALL]")

    ds.select("haversine")
      .withColumn("unified", lit(1))
      .map(haversine =>
        HaversineRoundedKeyValue(haversine.getInt(1).toString, math.round(haversine.getDouble(0)).toInt)
      )
      .withColumn("cume_dist", cume_dist().over(orderedDsPartitioner))
      .distinct()
  }

  def getLongestPathsPerDay(spark: SparkSession, ds:Dataset[HaversineDistanced], k: Int): Dataset[HaversineDistanced] = {
    import spark.implicits._

    ds
      .withColumn("rank", rank().over(reverseDsPartitioner))
      .filter(rankedHaversine => rankedHaversine.getInt(3) <= k)
      .select("dataset", "subscriber", "haversine")
      .as[HaversineDistanced]
  }

  def getLongestPathsTransformedData(spark: SparkSession, ds:Dataset[ReducedTelekomEvent], haversineDistancedArray: Array[HaversineDistanced]): Dataset[ReducedTelekomEvent] = {

    ds
      .filter( inverseFilter => longestPathFilterCriteria(haversineDistancedArray, inverseFilter) )
      .orderBy("dataset", "subscriber", "timestamp")
  }

  private def longestPathFilterCriteria(haversineArray: Array[HaversineDistanced], event: ReducedTelekomEvent): Boolean = {
    for (haversine <- haversineArray) {
      if (haversine.dataset == event.dataset && haversine.subscriber == event.subscriber) {
        return true
      }
    }
    false
  }

}
