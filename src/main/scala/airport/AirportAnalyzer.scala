package airport

import filter.CellFilter
import io.{ReducedTelekomEvent, TelekomId}
import org.apache.spark.sql.expressions.{Window, WindowSpec}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Dataset, SparkSession}

object AirportAnalyzer {

  val orderedDsPartitioner: WindowSpec = Window.partitionBy("dataset", "subscriber").orderBy(col("timestamp").desc)



  def getPathsEndingAtAirport(spark: SparkSession, ds: Dataset[ReducedTelekomEvent]): Dataset[ReducedTelekomEvent] = {
    import spark.implicits._

    val departingArray = ds
      .withColumn("order", row_number().over(orderedDsPartitioner))
      .filter($"order" === 1)
      .withColumn("distanceFromAirport", CellFilter.udfDistanceFromAirport(ds("lat"), ds("lng")))
      .filter($"distanceFromAirport" <= CellFilter.airportRadius)
      .select("dataset", "subscriber")
      .as[TelekomId]
      .collect()

    ds
      .filter(r => departingArray.contains(TelekomId(r.dataset, r.subscriber)))
      .orderBy("dataset", "subscriber", "timestamp")
  }
}
