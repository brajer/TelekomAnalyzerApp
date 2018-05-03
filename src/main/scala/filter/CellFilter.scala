package filter

import distance.Haversine
import io.{Cell, ReducedTelekomEvent}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Dataset, SparkSession}

object CellFilter {

  /* BUDAPEST */
  val outerBudapestCenter = Cell(47.49678009, 19.0786338)
  val outerBudapestRadius = 20

  val innerBudapestCenter = Cell(47.4963477, 19.0402499)
  val innerBudapestRadius = 2.5

  val udfDistanceFromOuterBudapest: UserDefinedFunction = udf(
    (lat: Double, lng: Double) => Haversine.calculateDistance(lat, lng, outerBudapestCenter.lat, outerBudapestCenter.lng)
  )

  val udfDistanceFromInnerBudapest: UserDefinedFunction = udf(
    (lat: Double, lng: Double) => Haversine.calculateDistance(lat, lng, innerBudapestCenter.lat, innerBudapestCenter.lng)
  )

  /* AIRPORT */
  val airport = Cell(47.4353966, 19.2344533)
  val airportRadius = 3 // in km

  val udfDistanceFromAirport: UserDefinedFunction = udf(
    (lat: Double, lng: Double) => Haversine.calculateDistance(lat, lng, airport.lat, airport.lng)
  )



  def withOuterBudapestCells(spark: SparkSession, ds:Dataset[ReducedTelekomEvent]): Dataset[ReducedTelekomEvent] = {
    import spark.implicits._

    ds
      .withColumn("distance", udfDistanceFromOuterBudapest(ds("lat"), ds("lng")))
      .filter($"distance" <= outerBudapestRadius)
      .drop($"distance")
      .as[ReducedTelekomEvent]
  }

  def withInnerBudapestCells(spark: SparkSession, ds:Dataset[ReducedTelekomEvent]): Dataset[ReducedTelekomEvent] = {
    import spark.implicits._

    ds
      .withColumn("distance", udfDistanceFromInnerBudapest(ds("lat"), ds("lng")))
      .filter($"distance" <= innerBudapestRadius)
      .drop($"distance")
      .as[ReducedTelekomEvent]
  }

  def withAirportCells(spark: SparkSession, ds:Dataset[ReducedTelekomEvent]): Dataset[ReducedTelekomEvent] = {
    import spark.implicits._

    ds
      .withColumn("distance", udfDistanceFromAirport(ds("lat"), ds("lng")))
      .filter($"distance" <= airportRadius)
      .drop($"distance")
      .as[ReducedTelekomEvent]
  }
}
