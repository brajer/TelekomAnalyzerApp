package io

import org.apache.spark.sql.{Dataset, SaveMode, SparkSession}

object DataIo {

  def readCsvTelekomData(spark: SparkSession, inputPath: String): Dataset[OriginalTelekomEvent] = {
    import spark.implicits._

    spark.read
      .option("header", "true")
      .option("charset", "UTF-8")
      .option("delimiter", ";")
      .option("inferSchema", "true")
      .csv(inputPath)
      .as[OriginalTelekomEvent]
  }

  def writeGenericJson(spark: SparkSession, ds: Dataset[_], outputPath: String): Unit = {
    ds
      .coalesce(1)
      .write
      //.mode(SaveMode.Append)
      .mode(SaveMode.Overwrite)
      .json(outputPath)
  }

  def writeGenericCsv(spark: SparkSession, ds: Dataset[_], outputPath: String): Unit = {
    ds
      //.repartition(1)
      .coalesce(1)
      .write
      .option("header", "true")
      .mode(SaveMode.Overwrite)
      .csv(outputPath)
  }
}
