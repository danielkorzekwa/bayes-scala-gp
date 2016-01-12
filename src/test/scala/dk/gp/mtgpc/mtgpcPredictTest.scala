package dk.gp.mtgpc

import org.junit.Assert.assertEquals
import org.junit.Test

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector

class mtgpcPredictTest {

  val (x, y) = getMtGpcTestData()

  val covFunc = TestMtGpcCovFunc()
  val covFuncParams = DenseVector(3.9167, 1.3032, 1.1069) //log sf, logEllx1, logEllx2
  val gpMean = -6.5124

  val xTest = DenseMatrix((1.0, 4.0, -4.0), (1.0, -2.5, -1.3), (2.0, 4.0, -4.0), (2.0, -2.5, -1.3), (3.0, 4.0, -4.0), (3.0, -2.5, -1.3))

  @Test def test = {

    val model = MtgpcModel(x, y, covFunc, covFuncParams, gpMean)
    val predicted = mtgpcPredict(xTest, model)

    assertEquals(0.125, predicted(0), 0.0001)
    assertEquals(0.4341, predicted(1), 0.0001)

    assertEquals(0.1564, predicted(2), 0.0001)
    assertEquals(0.9738, predicted(3), 0.0001)

    assertEquals(0.3899, predicted(4), 0.0001)
    assertEquals(0.2048, predicted(5), 0.0001)
  }
}