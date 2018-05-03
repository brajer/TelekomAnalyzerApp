import airport.AirportAnalyzer
import config.AppConfiguration
import crowd.{CrowdAnalyzer, CrowdPredictor}
import distance.DistanceAnalyzer
import filter.CellFilter
import io.DataTransform.withReducedTelekomData
import io.{DataIo, DataTransform}
import org.apache.spark.sql.SparkSession

object TelekomAnalyzerApp {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("TelekomAnalyzerApp")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    /* DATA PREPARATION AND PRUNING */
    println("Reading Telekom data...")
    val originalDs = DataIo.readCsvTelekomData(spark, AppConfiguration.InputPathPrefix + "/" + AppConfiguration.InputFilePattern)
    val ds = withReducedTelekomData(spark, originalDs).persist()

    val isFilteringActive = AppConfiguration.filterOuterBudapest ||
                            AppConfiguration.filterInnerBudapest ||
                            AppConfiguration.filterAirport

    /* DATA EXPLORATION & STATISTICS */
    if (AppConfiguration.ShowDatasetInfo) {
      println("Original Telekom data...")
      originalDs.describe().show()

      println("Reduced Telekom data...")
      ds.describe().show()
    }

    if (isFilteringActive) {
      val outerBudapestDs = CellFilter.withOuterBudapestCells(spark, ds)
      val innerBudapestDs = CellFilter.withInnerBudapestCells(spark, ds)
      val airportDs = CellFilter.withAirportCells(spark, ds)

      if (AppConfiguration.ShowDatasetInfo) {
        println("Outer Budapest data analysis...")
        outerBudapestDs.describe().show()

        println("Inner Budapest data analysis...")
        innerBudapestDs.describe().show()

        println("Airport data analysis...")
        airportDs.describe().show()
      }
    }

    /* UNIQUE CELL TOWERS */
    val uniqueCells = DataTransform.withUniqueCells(spark, ds).cache()
    if (AppConfiguration.RunCellExport) {
      println("[0] Writing unique cell tower data...")
      DataIo.writeGenericCsv(spark, uniqueCells, AppConfiguration.OutputPathPrefix + "/cells")
    }

    /* DISTANCES TRAVELLED PER DAY AND SUBSCRIBER */
    val uniqueVisits = DataTransform.withUniqueVisits(spark, ds).cache()
    if (AppConfiguration.RunDistancePerDayAnalyser) {
      println("[1] Preparing distance data...")
      val distancedDs = DataTransform.withDistanceTraveledByDayAndSubscriber(spark, uniqueVisits)
      val haversineKeyValueDs = DataTransform.withHaversineKeyValue(spark, distancedDs)

      println("[1] Analyzing distance data...")
      val distanceMetaData = DistanceAnalyzer.getDistanceMetaDataPerDay(spark, haversineKeyValueDs)
      val distanceQuantiles = DistanceAnalyzer.getDistanceQuantilesPerDay(spark, haversineKeyValueDs)
      val distanceCumeDist = DistanceAnalyzer.getDistanceCumulativeDistributionPerDay(spark, haversineKeyValueDs)

      if (AppConfiguration.RunDistancePerDayExport) {
        println("[1] Writing distance data...")
        DataIo.writeGenericCsv(spark, distanceMetaData, AppConfiguration.OutputPathPrefix + "/metadata")
        DataIo.writeGenericCsv(spark, distanceQuantiles, AppConfiguration.OutputPathPrefix + "/quantiles")
        DataIo.writeGenericCsv(spark, distanceCumeDist, AppConfiguration.OutputPathPrefix + "/cume_dist")
      }
    }

    /* LONGEST k PATHS PER DAY */
    if (AppConfiguration.RunLongestPathsPerDayAnalyser) {
      val distancedDs = DataTransform.withDistanceTraveledByDayAndSubscriber(spark, uniqueVisits)

      println("[2] Analyzing %d longest paths per day".format(AppConfiguration.longestPaths))
      val longestPathPerDay = DistanceAnalyzer.getLongestPathsPerDay(spark, distancedDs, AppConfiguration.longestPaths)

      println("[2] Extract data for longest paths")
      val longestPathsArray = longestPathPerDay.collect()
      val longestPathsDs = DistanceAnalyzer.getLongestPathsTransformedData(spark, ds, longestPathsArray)

      if (AppConfiguration.RunLongestPathsPerDayExport) {
        println("[2] Writing longest paths data...")
        DataIo.writeGenericCsv(spark, longestPathsDs, AppConfiguration.OutputPathPrefix + "/longest_paths")
      }
    }
    uniqueVisits.unpersist()

    /* AIRPORT ANALYZER */
    if (AppConfiguration.RunAirportAnalyzer) {
      println("[3] Running Airport analyzer...")
      val departingPaths = AirportAnalyzer.getPathsEndingAtAirport(spark, DataTransform.withSingletonEventsFiltered(spark, ds))

      if (AppConfiguration.RunAirportExport) {
        println("[3] Writing Airport data...")
        DataIo.writeGenericCsv(spark, departingPaths, AppConfiguration.OutputPathPrefix + "/airport")
      }
    }

    /* CROWD ANALYZER */
    if (AppConfiguration.RunCrowdAnalyzer) {
      println("[4] Running Crowd analyzer...")
      val timeSeriesDs = DataTransform.withTimeSeriesFrames(spark, ds)
      val timeSeriesCrowdDs = CrowdAnalyzer.getCellCrowdTimeSeries(spark, timeSeriesDs)
      //DataAnalyzer.analyzeGenericData(timeSeriesCrowdDs)

      if (AppConfiguration.RunCrowdExport) {
        println("[4] Writing Crowd data...")
        DataIo.writeGenericCsv(spark, timeSeriesCrowdDs, AppConfiguration.OutputPathPrefix + "/crowd")
      }
    }

    /* CROWD PREDICTION */
    if (AppConfiguration.RunKnnPredictor) {
      println("[5] Running KNN analyzer...")
      val timeSeriesDs = DataTransform.withTimeSeriesFrames(spark, ds)
      val timeSeriesCrowdDs = CrowdAnalyzer.getCellCrowdTimeSeries(spark, timeSeriesDs).cache()

      val knnCells = DataTransform.withKNNCells(spark, uniqueCells, AppConfiguration.kNeighbours).cache()
      val knnPredictionDs = CrowdPredictor.getKnnCrowdPredictionData(spark, knnCells, timeSeriesCrowdDs)

      CrowdPredictor.makeCrowdPredictionWithLogisticRegression(spark, knnPredictionDs)
      CrowdPredictor.makeCrowdPredictionWithLinearRegression(spark, knnPredictionDs)
      CrowdPredictor.makeCrowdPredictionWithGLR(spark, knnPredictionDs)
      CrowdPredictor.makeCrowdPredictionWithNaiveBayes(spark, knnPredictionDs)
      CrowdPredictor.makeCrowdPredictionWithDecisionTreeClassifier(spark, knnPredictionDs)
      CrowdPredictor.makeCrowdPredictionWithDecisionTreeRegression(spark, knnPredictionDs)

      if (AppConfiguration.RunKnnPredictionExport) {
      }
    }

    spark.stop()
  }
}
