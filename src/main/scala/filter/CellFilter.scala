package filter

import distance.Haversine
import io.{Cell, ReducedTelekomEvent}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Dataset, SparkSession}

object CellFilter {

  /* BUDAPEST */
  val outerBudapestCenter = Cell(47.49678009, 19.0786338)
  val outerBudapestRadius = 20

  val innerBudapestCenter = Cell(47.4963477, 19.0402499)
  val innerBudapestRadius = 2.5

  val distanceFromOuterBudapest = (lat: Double, lng: Double) => { Haversine.calculateDistance(lat, lng, outerBudapestCenter.lat, outerBudapestCenter.lng) }
  val sqlDistanceFromOuterBudapest = udf(distanceFromOuterBudapest)

  val distanceFromInnerBudapest = (lat: Double, lng: Double) => { Haversine.calculateDistance(lat, lng, innerBudapestCenter.lat, innerBudapestCenter.lng) }
  val sqlDistanceFromInnerBudapest = udf(distanceFromInnerBudapest)

  /* AIRPORT */
  val airport = Cell(47.4288208, 19.2547013)
  val airportRadius = 1

  val distanceFromAirport = (lat: Double, lng: Double) => { Haversine.calculateDistance(lat, lng, airport.lat, airport.lng) }
  val sqlDistanceFromAirport = udf(distanceFromAirport)



  def withOuterBudapestCells(spark: SparkSession, ds:Dataset[ReducedTelekomEvent]): Dataset[ReducedTelekomEvent] = {
    import spark.implicits._

    ds
      .withColumn("distance", sqlDistanceFromOuterBudapest(ds("lat"), ds("lng")))
      .filter($"distance" <= outerBudapestRadius)
      .drop($"distance")
      .as[ReducedTelekomEvent]
  }

  def withInnerBudapestCells(spark: SparkSession, ds:Dataset[ReducedTelekomEvent]): Dataset[ReducedTelekomEvent] = {
    import spark.implicits._

    ds
      .withColumn("distance", sqlDistanceFromInnerBudapest(ds("lat"), ds("lng")))
      .filter($"distance" <= innerBudapestRadius)
      .drop($"distance")
      .as[ReducedTelekomEvent]
  }

  def withAirportCells(spark: SparkSession, ds:Dataset[ReducedTelekomEvent]): Dataset[ReducedTelekomEvent] = {
    import spark.implicits._

    ds
      .withColumn("distance", sqlDistanceFromAirport(ds("lat"), ds("lng")))
      .filter($"distance" <= airportRadius)
      .drop($"distance")
      .as[ReducedTelekomEvent]
  }

}
