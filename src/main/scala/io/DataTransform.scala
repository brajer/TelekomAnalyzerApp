package io

import java.sql.Timestamp

import config.AppConfiguration
import distance.Haversine
import filter.CellFilter
import org.apache.spark.sql.expressions.{UserDefinedFunction, Window, WindowSpec}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import util.TimeConverter

final case class TelekomId(dataset: String, subscriber: String)
final case class Cell(lat: Double, lng: Double)
final case class OriginalTelekomEvent(callId: Int, subscriber: String, ni: String, dt: String, TAC: String, eventType: Int, timestamp: Timestamp, cellId: String, lat: Double, lng: Double)
final case class ReducedTelekomEvent(dataset: String, subscriber: String, timestamp: Timestamp, lat: Double, lng: Double)
final case class UniqueVisits(dataset: String, subscriber: String, lat: Double, lng: Double)
final case class HaversineDistanced(dataset: String, subscriber: String, haversine: Double)
final case class HaversineKeyValue(dataset: String, haversine:Double)
final case class TimeSeriesTelekomEvent(dataset: String, dataframe: Int, lat: Double, lng: Double)
final case class TimeSeriesGroupedTelekomEvent(dataset: String, dataframe: Int, lat: Double, lng: Double, crowd: BigInt)
final case class PairedCellDistances(lat: Double, lng: Double, oLat: Double, oLng: Double, haversine: Double)

object DataTransform {

  val udfTimeConverter: UserDefinedFunction = udf((timestamp: Timestamp) => TimeConverter.convertTimestampToDate(timestamp))
  val udfGetHour: UserDefinedFunction = udf((timestamp: Timestamp) => timestamp.toLocalDateTime.getHour)

  val dsGroupingCriteria: WindowSpec = Window.partitionBy("dataset", "subscriber")
  val dsDistanceRankingCriteria: WindowSpec = Window.partitionBy("lat", "lng").orderBy("haversine")



  def withReducedTelekomData(spark: SparkSession, ds: Dataset[OriginalTelekomEvent]): Dataset[ReducedTelekomEvent] = {
    import spark.implicits._

    val unfilteredDs = ds
      .withColumn("dataset", udfTimeConverter(ds("timestamp")))
      .select("dataset", "subscriber", "timestamp", "lat", "lng")
      .filter($"lat" > 0.0 && $"lng" > 0.0)
      .na.drop()
      .as[ReducedTelekomEvent]

    if (AppConfiguration.filterOuterBudapest) {
      return CellFilter.withOuterBudapestCells(spark, unfilteredDs)
    }
    if (AppConfiguration.filterInnerBudapest) {
      return CellFilter.withInnerBudapestCells(spark, unfilteredDs)
    }
    if (AppConfiguration.filterAirport) {
      return CellFilter.withAirportCells(spark, unfilteredDs)
    }

    unfilteredDs
  }

  def withUniqueCells(spark: SparkSession, ds: Dataset[_]): Dataset[Cell] = {
    import spark.implicits._

    ds
      .select("lat", "lng")
      .dropDuplicates()
      .as[Cell]
  }

  def withSingletonEventsFiltered(spark: SparkSession, ds: Dataset[ReducedTelekomEvent]): Dataset[ReducedTelekomEvent] = {
    import spark.implicits._

    ds
      .withColumn("count", count("timestamp").over(dsGroupingCriteria))
      .filter($"count" > 1)
      .drop(col("count"))
      .as[ReducedTelekomEvent]
  }

  def withUniqueVisits(spark: SparkSession, ds: Dataset[ReducedTelekomEvent]): Dataset[UniqueVisits] = {
    import spark.implicits._

    ds
      .select("dataset", "subscriber", "lat", "lng")
      .distinct()
      .as[UniqueVisits]
  }

  def withDistanceTraveledByDayAndSubscriber(spark: SparkSession, ds: Dataset[UniqueVisits]): Dataset[HaversineDistanced] = {
    import spark.implicits._

    ds.join(ds, Seq("dataset", "subscriber"))
      .distinct()
      .map{ case Row(dataset: String, subscriber: String, lat1: Double, lng1: Double, lat2: Double, lng2: Double) =>
        HaversineDistanced(dataset, subscriber, Haversine.calculateDistance(lat1, lng1, lat2, lng2)) }
      .withColumn("haversine", round($"haversine", 2))
      .groupBy("dataset", "subscriber") // Catalyst Optimized?
      .max("haversine")
      .withColumnRenamed("max(haversine)", "haversine")
      .as[HaversineDistanced]
  }

  def withHaversineKeyValue(spark: SparkSession, ds:Dataset[HaversineDistanced]): Dataset[HaversineKeyValue] = {
    import spark.implicits._

    ds
      .select("dataset", "haversine")
      .as[HaversineKeyValue]
  }

  def withTimeSeriesFrames(spark: SparkSession, ds:Dataset[ReducedTelekomEvent]): Dataset[TimeSeriesTelekomEvent] = {
    import spark.implicits._

    ds.select("dataset", "timestamp", "lat", "lng")
      .withColumn("dataframe", udfGetHour(ds("timestamp")))
      .map{ case Row(dataset: String, timestamp: Timestamp, lat: Double, lng: Double, dataframe: Int) =>
        TimeSeriesTelekomEvent(dataset, dataframe, lat, lng)
      }
      .as[TimeSeriesTelekomEvent]
  }

  def withKNNCells(spark: SparkSession, ds: Dataset[Cell], k: Int): Dataset[PairedCellDistances] = {
    import spark.implicits._

    println("...Getting KNN cells")

    ds
      .crossJoin(ds)
      .filter(r => !(r.getDouble(0) == r.getDouble(2) && r.getDouble(1) == r.getDouble(3))) // filter self relations
      .map{ case Row(lat: Double, lng: Double, oLat: Double, oLng: Double) =>
        PairedCellDistances(lat, lng, oLat, oLng, Haversine.calculateDistance(lat, lng, oLat, oLng)) }
      .withColumn("haversine", round($"haversine", 2))
      .withColumn("nearness", rank().over(dsDistanceRankingCriteria))
      .filter(r => r.getInt(5) <= k)
      .select("lat", "lng", "oLat", "oLng", "haversine")
      .as[PairedCellDistances]
  }
}
