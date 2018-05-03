package crowd

import io.{TimeSeriesGroupedTelekomEvent, TimeSeriesTelekomEvent}
import org.apache.spark.sql.{Dataset, SparkSession}

object CrowdAnalyzer {

  def getCellCrowdTimeSeries(spark: SparkSession, ds: Dataset[TimeSeriesTelekomEvent]): Dataset[TimeSeriesGroupedTelekomEvent] = {
    import spark.implicits._

    ds
      .groupBy("dataset", "dataframe", "lat", "lng")
      .count()
      .orderBy("dataset", "dataframe")
      .withColumnRenamed("count", "crowd")
      .as[TimeSeriesGroupedTelekomEvent]
  }
}
