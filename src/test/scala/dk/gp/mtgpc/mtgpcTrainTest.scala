package dk.gp.mtgpc

import org.junit._
import Assert._
import breeze.linalg.DenseVector
import breeze.numerics.log

class mtgpcTrainTest {

  val (x, y) = getMtGpcTestData()
  val covFunc = TestMtGpcCovFunc()
  val covFuncParams = DenseVector(log(1), log(1), log(1)) //log sf, logEllx1, logEllx2
  val gpMean = 0

  @Test def test = {

    val initialModel = MtgpcModel(x, y, covFunc, covFuncParams, gpMean)
    val trainedModel = mtgpcTrain(initialModel, maxIter = 11)
    // assertEquals(-6.3811, trainedModel.gpMean, 0.0001)

    // assertEquals(4.0346, trainedModel.covFuncParams(0), 0.0001) //logSf
    // assertEquals(1.4803, trainedModel.covFuncParams(1), 0.0001) //logEllx1
    //  assertEquals(1.48042, trainedModel.covFuncParams(2), 0.0001) //logEllx2

    val initialLoglik = MtGpcDiffFunction(initialModel).calculate(DenseVector(initialModel.covFuncParams.toArray :+ initialModel.gpMean))._1
    assertEquals(48.95021, initialLoglik, 0.0001)
    val loglik = MtGpcDiffFunction(trainedModel).calculate(DenseVector(trainedModel.covFuncParams.toArray :+ trainedModel.gpMean))._1
    assertEquals(39.8589, loglik, 1)
  }
}