package crowd

import io.{PairedCellDistances, TimeSeriesGroupedTelekomEvent}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier, LogisticRegression, LogisticRegressionModel, NaiveBayes, NaiveBayesModel}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, DecisionTreeRegressor, GeneralizedLinearRegression, GeneralizedLinearRegressionModel, LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.expressions.{UserDefinedFunction, Window, WindowSpec}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

final case class KnnPredictionReady(dataset: String, dataframe: Int, lat: Double, lng: Double, crowd: BigInt, lastFrameCrowd: BigInt, knnMin: Double, knnSum: Double, knnAvg: Double, knnClosest: Double)

object CrowdPredictor {

  val udfWithWeightedCrowd: UserDefinedFunction = udf(
    (haversine: Double, crowd: java.math.BigDecimal) => crowd.doubleValue()/haversine
  )

  val timeseriesDsOrderer: WindowSpec = Window.orderBy("dataset", "dataframe", "lat", "lng")
  val timeseriesDsPartitioner: WindowSpec = Window.partitionBy("dataset", "dataframe", "lat", "lng")

  val features = Array("datasetIndex", "dataframe", "lat", "lng", "lastFrameCrowd", "knnMin", "knnSum", "knnAvg", "knnClosest")
  val featuresNoLatLng = Array("datasetIndex", "dataframe", "lastFrameCrowd", "knnMin", "knnSum", "knnAvg", "knnClosest")

  val indexer: StringIndexer = new StringIndexer()
    .setInputCol("dataset")
    .setOutputCol("datasetIndex")

  /* Feature vectorizer */
  val assembler: VectorAssembler = new VectorAssembler()
    .setInputCols(features)
    .setOutputCol("features")



  def getKnnCrowdPredictionData(spark: SparkSession, knnCells: Dataset[PairedCellDistances], timeseriesDs: Dataset[TimeSeriesGroupedTelekomEvent]): Dataset[KnnPredictionReady] = {
    import spark.implicits._

    val oTimeseriesDs = timeseriesDs
                          .withColumnRenamed("lat", "oLat")
                          .withColumnRenamed("lng", "oLng")
                          .withColumnRenamed("crowd", "oCrowd")

    timeseriesDs
      .withColumn("lastFrameCrowd", lag("crowd", 1, 0).over(timeseriesDsOrderer))
      .join(knnCells, Seq("lat", "lng"))
      .join(oTimeseriesDs, Seq("dataset", "dataframe", "oLat", "oLng"))
      //.withColumn("weightedCrowd", sqlWithWeightedCrowd(knnCells("haversine"), oTimeseriesDs("oCrowd")))
      //.withColumn("weightedCrowd", round($"weightedCrowd", 2))
      .withColumn("knnMin", min("oCrowd").over(timeseriesDsPartitioner))
      .withColumn("knnSum", round(sum("oCrowd").over(timeseriesDsPartitioner), 2))
      .withColumn("knnAvg", round(avg("oCrowd").over(timeseriesDsPartitioner), 2))
      .withColumn("knnClosest", round(max("haversine").over(timeseriesDsPartitioner), 2))
      //.withColumn("knnStddev", stddev("weightedCrowd").over(timeseriesDsPartitioner))
      .select("dataset", "dataframe", "lat", "lng", "crowd", "lastFrameCrowd", "knnMin", "knnSum", "knnAvg", "knnClosest")
      .as[KnnPredictionReady]
  }

  def makeCrowdPredictionWithLogisticRegression(spark: SparkSession, predDs: Dataset[KnnPredictionReady]): Unit = {

    println("...Running Logistic Regression")

    val crowdIndexer = new StringIndexer()
      .setInputCol("crowd")
      .setOutputCol("crowdIndex")
      .fit(predDs)

    /* Split data to training and test sets */
    val Array(training, test) = predDs.randomSplit(Array(0.7, 0.3), seed = 27L)

    /* Logistic Regression Model */
     val lr = new LogisticRegression()
                      .setFeaturesCol("features")
                      .setLabelCol("crowdIndex")
                      .setMaxIter(10)

    //val lrModel = lr.fit(featuredDs)
    /* Create pipeline */
    val pipeline = new Pipeline()
                    .setStages(Array(crowdIndexer, indexer, assembler, lr))

    //val lrModel = pipeline.fit(predDs)
    /* Create regression evaluator */
    val rmseEvaluator = new RegressionEvaluator()
                          .setLabelCol("crowdIndex")
                          .setMetricName("rmse") // "rmse": root mean squred error, "mse": mean squared error, "mae": mean absolute error, "r2": r-squared

    /* Create parameter grid for parameter hyper-tuning */
    val paramGrid = new ParamGridBuilder()
                          .addGrid(lr.regParam, Array(0.01, 0.1))
                          .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
                          .build()

    val cv = new CrossValidator()
                  .setEstimator(pipeline)
                  .setEvaluator(rmseEvaluator)
                  .setEstimatorParamMaps(paramGrid)
                  .setNumFolds(4)


    val cvModel = cv.fit(training)

    /* Extract best model from Cross-validation and hyper-parameter tuning */
    val lrModel: LogisticRegressionModel = cvModel.bestModel
                                                .asInstanceOf[PipelineModel]
                                                .stages
                                                .last.asInstanceOf[LogisticRegressionModel]

    println(s"Multinomial coefficients: ${lrModel.coefficientMatrix}")
    println(s"Multinomial intercepts: ${lrModel.interceptVector}")

    val predictions = cvModel.transform(test)

    getEvaluationMetrics(predictions, "crowdIndex")
  }

  def makeCrowdPredictionWithLinearRegression(spark: SparkSession, predDs: Dataset[KnnPredictionReady]): Unit = {

    println("...Running Linear Regression")

    /* Split data to training and test sets */
    val Array(training, test) = predDs.randomSplit(Array(0.7, 0.3), seed = 27L)

    /* Linear Regression Model */
    val lr = new LinearRegression()
                    .setFeaturesCol("features")
                    .setLabelCol("crowd")
                    .setMaxIter(10)

    val pipeline = new Pipeline()
      .setStages(Array(indexer, assembler, lr))

    val regressionEvaluator = new RegressionEvaluator()
      .setLabelCol("crowd")
      .setMetricName("rmse")

    /* Create parameter grid for parameter hyper-tuning */
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.01, 0.1))
      .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 0.8, 1.0))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(regressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(4)

    val cvModel = cv.fit(training)

    /* Extract best model from Cross-validation and hyper-parameter tuning */
    val lrModel: LinearRegressionModel = cvModel.bestModel
      .asInstanceOf[PipelineModel]
      .stages
      .last.asInstanceOf[LinearRegressionModel]

    // Summarize the model over the training set and print out some metrics
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
    val trainingSummary = lrModel.summary
    trainingSummary.residuals.show()

    val predictions = cvModel.transform(test)

    getEvaluationMetrics(predictions, "crowd")
  }

  def makeCrowdPredictionWithGLR(spark: SparkSession, predDs: Dataset[KnnPredictionReady]): Unit = {

    val family = "poisson" // "gaussian", "binomial", "poisson", "gamma"

    println("...Running Generalized Linear Regression with '" + family + "' distribution")

    /* Split data to training and test sets */
    val Array(training, test) = predDs.randomSplit(Array(0.7, 0.3), seed = 27L)

    /* GLM Poisson */
    val glr = new GeneralizedLinearRegression()
                    .setFamily(family)
                    .setFeaturesCol("features")
                    .setLabelCol("crowd")
                    .setMaxIter(10)

    val pipeline = new Pipeline()
      .setStages(Array(indexer, assembler, glr))

    val regressionEvaluator = new RegressionEvaluator()
      .setLabelCol("crowd")
      .setMetricName("rmse")

    /* Create parameter grid for parameter hyper-tuning */
    val paramGrid = new ParamGridBuilder()
      .addGrid(glr.regParam, Array(0.01, 0.1))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(regressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(4)

    val cvModel = cv.fit(training)

    /* Extract best model from Cross-validation and hyper-parameter tuning */
    val glrModel: GeneralizedLinearRegressionModel = cvModel.bestModel
      .asInstanceOf[PipelineModel]
      .stages
      .last.asInstanceOf[GeneralizedLinearRegressionModel]

    // Print the coefficients and intercept for generalized linear regression model
    println(s"Coefficients: ${glrModel.coefficients}")
    println(s"Intercept: ${glrModel.intercept}")

    // Summarize the model over the training set and print out some metrics
    val summary = glrModel.summary
    println(s"Coefficient Standard Errors: ${summary.coefficientStandardErrors.mkString(",")}")
    println(s"T Values: ${summary.tValues.mkString(",")}")
    println(s"P Values: ${summary.pValues.mkString(",")}")
    println(s"Residual Degree Of Freedom Null: ${summary.residualDegreeOfFreedomNull}")
    println(s"Deviance: ${summary.deviance}")
    println(s"Residual Degree Of Freedom: ${summary.residualDegreeOfFreedom}")
    println(s"AIC: ${summary.aic}")
    println("Deviance Residuals: ")
    summary.residuals().show()

    val predictions = cvModel.transform(test)

    getEvaluationMetrics(predictions, "crowd")
  }

  def makeCrowdPredictionWithNaiveBayes(spark: SparkSession, predDs: Dataset[KnnPredictionReady]): Unit = {

    println("...Running Naive Bayes")

    val crowdIndexer = new StringIndexer()
      .setInputCol("crowd")
      .setOutputCol("crowdIndex")
      .fit(predDs)

    /* Split data to training and test sets */
    val Array(training, test) = predDs.randomSplit(Array(0.7, 0.3), seed = 27L)

    /* Linear Regression Model */
    val nb = new NaiveBayes()
      .setFeaturesCol("features")
      .setLabelCol("crowdIndex")

    val pipeline = new Pipeline()
      .setStages(Array(crowdIndexer, indexer, assembler, nb))

    val multiClassAccuracyEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("crowdIndex")
      .setMetricName("accuracy")

    /* Create parameter grid for parameter hyper-tuning */
    val paramGrid = new ParamGridBuilder().build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(multiClassAccuracyEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(4)

    val cvModel = cv.fit(training)

    /* Extract best model from Cross-validation and hyper-parameter tuning */
    val nbModel: NaiveBayesModel = cvModel.bestModel
      .asInstanceOf[PipelineModel]
      .stages
      .last.asInstanceOf[NaiveBayesModel]

    println(s"Pi: ${nbModel.pi}")
    println(s"Theta: ${nbModel.theta}")

    val predictions = cvModel.transform(test)

    getEvaluationMetrics(predictions, "crowdIndex")
  }

  def makeCrowdPredictionWithDecisionTreeClassifier(spark: SparkSession, predDs: Dataset[KnnPredictionReady]): Unit = {

    println("...Running Decision Tree")

    val crowdIndexer = new StringIndexer()
      .setInputCol("crowd")
      .setOutputCol("crowdIndex")
      .fit(predDs)

    /* Split data to training and test sets */
    val Array(training, test) = predDs.randomSplit(Array(0.7, 0.3), seed = 27L)

    /* Linear Regression Model */
    val dt = new DecisionTreeClassifier()
      .setFeaturesCol("features")
      .setLabelCol("crowdIndex")

/*    val crowdConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(crowdIndexer.labels)*/

    val pipeline = new Pipeline()
      .setStages(Array(crowdIndexer, indexer, assembler, dt))

    val multiClassAccuracyEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("crowdIndex")
      .setMetricName("accuracy")

    /* Create parameter grid for parameter hyper-tuning */
    val paramGrid = new ParamGridBuilder()
      .addGrid(dt.maxDepth, Array(5, 10))
      .addGrid(dt.minInstancesPerNode, Array(1, 2, 4))
      .addGrid(dt.impurity, Array("gini"))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(multiClassAccuracyEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(4)

    val cvModel = cv.fit(training)

    /* Extract best model from Cross-validation and hyper-parameter tuning */
    val treeModel: DecisionTreeClassificationModel = cvModel.bestModel
      .asInstanceOf[PipelineModel]
      .stages
      .last.asInstanceOf[DecisionTreeClassificationModel]

    //println("Learned classification tree model:\n" + treeModel.toDebugString)
    println(s"Feature importances: ${treeModel.featureImportances}")

    val predictions = cvModel.transform(test)

    getEvaluationMetrics(predictions, "crowdIndex")
  }

  def makeCrowdPredictionWithDecisionTreeRegression(spark: SparkSession, predDs: Dataset[KnnPredictionReady]): Unit = {

    println("...Running Decision Tree Regression")

    /* Split data to training and test sets */
    val Array(training, test) = predDs.randomSplit(Array(0.7, 0.3), seed = 27L)

    /* Linear Regression Model */
    val dt = new DecisionTreeRegressor()
      .setFeaturesCol("features")
      .setLabelCol("crowd")

    val pipeline = new Pipeline()
      .setStages(Array(indexer, assembler, dt))

    val regressionEvaluator = new RegressionEvaluator()
      .setLabelCol("crowd")
      .setMetricName("rmse")

    /* Create parameter grid for parameter hyper-tuning */
    val paramGrid = new ParamGridBuilder()
      .addGrid(dt.maxDepth, Array(5, 10))
      .addGrid(dt.minInstancesPerNode, Array(1, 2, 4))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(regressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(4)

    val cvModel = cv.fit(training)

    /* Extract best model from Cross-validation and hyper-parameter tuning */
    val treeModel: DecisionTreeRegressionModel = cvModel.bestModel
      .asInstanceOf[PipelineModel]
      .stages
      .last.asInstanceOf[DecisionTreeRegressionModel]

    //println("Learned classification tree model:\n" + treeModel.toDebugString)
    println(s"Feature importances: ${treeModel.featureImportances}")

    val predictions = cvModel.transform(test)

    getEvaluationMetrics(predictions, "crowd")
  }


  private def getEvaluationMetrics(predictions: DataFrame, label: String): Unit = {

    val regressionRmseEvaluator = new RegressionEvaluator()
      .setLabelCol(label)
      .setMetricName("rmse")
    println("RMSE -> " + regressionRmseEvaluator.evaluate(predictions))

    val regressionMaeEvaluator = new RegressionEvaluator()
      .setLabelCol(label)
      .setMetricName("mae")
    println("MAE -> " + regressionMaeEvaluator.evaluate(predictions))

    val regressionR2Evaluator = new RegressionEvaluator()
      .setLabelCol(label)
      .setMetricName("r2")
    println("R2 -> " + regressionR2Evaluator.evaluate(predictions))


    val multiClassF1Evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol(label)
      .setMetricName("f1")
    println("F1 -> " + multiClassF1Evaluator.evaluate(predictions))

    val multiClassAccuracyEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol(label)
      .setMetricName("accuracy")
    println("Accuracy -> " + multiClassAccuracyEvaluator.evaluate(predictions))

    val multiClassPrecisionEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol(label)
      .setMetricName("weightedPrecision")
    println("Precision -> " + multiClassPrecisionEvaluator.evaluate(predictions))

/*    val multiClassRecallEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("crowd")
      .setMetricName("weightedRecall")
    println("Recall -> " + multiClassRecallEvaluator.evaluate(predictions))*/

  }
}
