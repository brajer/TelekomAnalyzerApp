package airport

import distance.Haversine
import io.{TelekomId, ReducedTelekomEvent}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Dataset, SparkSession}

object AirportAnalyzer {

  val airportLat = 47.4353966
  val airportLng = 19.2344533
  val airportRadius = 3 // in km

  val orderedDsPartitioner = Window.partitionBy("dataset", "subscriber").orderBy(col("timestamp").desc)

  val distanceFromAirport = (lat: Double, lng: Double) => { Haversine.calculateDistance(lat, lng, airportLat, airportLng) }
  val sqlDistanceFromAirport = udf(distanceFromAirport)



  def getPathsEndingAtAirport(spark: SparkSession, ds: Dataset[ReducedTelekomEvent]): Dataset[ReducedTelekomEvent] = {
    import spark.implicits._

    val departingArray = ds
      .withColumn("order", row_number().over(orderedDsPartitioner))
      .filter($"order" === 1)
      .withColumn("distanceFromAirport", sqlDistanceFromAirport(ds("lat"), ds("lng")))
      .filter($"distanceFromAirport" <= airportRadius)
      .select("dataset", "subscriber")
      .as[TelekomId]
      .collect()

    ds
      .filter(r => departingArray.contains(TelekomId(r.dataset, r.subscriber)))
      .orderBy("dataset", "subscriber", "timestamp")
  }

}
