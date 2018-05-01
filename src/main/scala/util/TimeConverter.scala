package util

import java.sql.Timestamp
import java.text.SimpleDateFormat

object TimeConverter {

  def convertTimestampToDate(timestamp: Timestamp) : String = {
    new SimpleDateFormat("MM.dd").format(timestamp)
  }
}