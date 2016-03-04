package dk.gp.hgpr

case class HgprAccuracy(
    allDataSize: Int, allRMSE: Double, allR2: Double,
    looCVRMSE:Double,looCVR2:Double,
    trainDataSize: Int, trainRMSE: Double, trainR2: Double,
    testDataSize: Int, testRMSE: Double, testR2: Double) {

  def toReport(): String = {
    val allReport = "All: n=%2d, rmse=%.3f, r2=%.3f".format(allDataSize, allRMSE, allR2)

    val looCVReport = "LooCV: n=%2d, rmse=%.3f, r2=%.3f".format(allDataSize, looCVRMSE, looCVR2)

    val trainReport = "Train: n=%2d, rmse=%.3f, r2=%.3f".format(trainDataSize, trainRMSE, trainR2)

    val testReport = "Test: n=%2d, rmse=%.3f, r2=%.3f".format(testDataSize, testRMSE, testR2)

    allReport + "\n" + looCVReport + "\n" + trainReport + "\n" + testReport
  }
}