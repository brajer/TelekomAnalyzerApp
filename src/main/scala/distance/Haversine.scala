package distance

import math._

object Haversine {

  val EARTH_RADIUS = 6371 // (km) 3956 in miles

  def calculateDistance(lat1: Double, lng1:Double, lat2:Double, lng2:Double): Double = {
    val dLat=(lat2 - lat1).toRadians
    val dLon=(lng2 - lng1).toRadians

    val a = pow(sin(dLat/2),2) + pow(sin(dLon/2),2) * cos(lat1.toRadians) * cos(lat2.toRadians)
    val c = 2 * asin(sqrt(a))

    EARTH_RADIUS * c
    //return BigDecimal(EARTH_RADIUS * c).setScale(precision, BigDecimal.RoundingMode.HALF_UP).toDouble
  }

}
