package config

object AppConfiguration {

  /* Data configuration */
  val InputFilePattern = "*_msc_*.gps"
  val InputPathPrefix  = "#input_path"
  val OutputPathPrefix = "#output_path"

  /* Area-based filter configuration */
  val filterOuterBudapest = false
  val filterInnerBudapest = false
  val filterAirport = false

  /* Analyzer configuration */
  val ShowDatasetInfo = true

  val RunCellExport = true

  val RunDistancePerDayAnalyser = true
  val RunDistancePerDayExport = true // depends on analyzer

  val RunLongestPathsPerDayAnalyser = true
  val RunLongestPathsPerDayExport = true // depends on analyzer
  val longestPaths = 5 // k longest paths per day

  val RunAirportAnalyzer = true
  val RunAirportExport = true // depends on analyzer

  val RunCrowdAnalyzer = true
  val RunCrowdExport = true // depends on analyzer

  val RunKnnPredictor = true
  val RunKnnPredictionExport = true // depends on predictor
  val kNeighbours = 3
}
