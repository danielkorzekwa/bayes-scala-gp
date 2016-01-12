package dk.gp.mtgpc

import org.junit.Assert.assertEquals
import org.junit.Test

import breeze.linalg.DenseVector
import breeze.numerics.log

class mtgpcTrainTest {

  val (x, y) = getMtGpcTestData()
  val covFunc = TestMtGpcCovFunc()
  val covFuncParams = DenseVector(log(1), log(1), log(1)) //log sf, logEllx1, logEllx2
  val gpMean = 0

  @Test def test = {

    val initialModel = MtgpcModel(x, y, covFunc, covFuncParams, gpMean)
    val trainedModel = mtgpcTrain(initialModel)
    assertEquals(-6.5124, trainedModel.gpMean, 0.0001)

    assertEquals(3.9167, trainedModel.covFuncParams(0), 0.0001) //logSf
    assertEquals(1.3032, trainedModel.covFuncParams(1), 0.0001) //logEllx1
    assertEquals(1.1069, trainedModel.covFuncParams(2), 0.0001) //logEllx2
  }
}