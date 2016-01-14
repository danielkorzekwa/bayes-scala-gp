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
    val trainedModel = mtgpcTrain(initialModel,maxIter=10)
    assertEquals(-6.3811, trainedModel.gpMean, 0.0001)

    assertEquals(4.1296, trainedModel.covFuncParams(0), 0.0001) //logSf
    assertEquals(1.3718, trainedModel.covFuncParams(1), 0.0001) //logEllx1
    assertEquals(1.32032, trainedModel.covFuncParams(2), 0.0001) //logEllx2
  }
}