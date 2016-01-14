package dk.gp.mtgpc

import org.junit.Assert.assertEquals
import org.junit.Test

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector

class mtgpcPredictTest {

  val (x, y) = getMtGpcTestData()

  val covFunc = TestMtGpcCovFunc()
  val covFuncParams = DenseVector(4.1296, 1.3718, 1.32032) //log sf, logEllx1, logEllx2
  val gpMean = -6.3811

  val xTest = DenseMatrix((1.0, 4.0, -4.0), (1.0, -2.5, -1.3), (2.0, 4.0, -4.0), (2.0, -2.5, -1.3), (3.0, 4.0, -4.0), (3.0, -2.5, -1.3))

  @Test def test = {

    val model = MtgpcModel(x, y, covFunc, covFuncParams, gpMean)
    val predicted = mtgpcPredict(xTest, model)

    assertEquals(0.0630, predicted(0), 0.0001)
    assertEquals(0.43137, predicted(1), 0.0001)

    assertEquals(0.11223, predicted(2), 0.0001)
    assertEquals(0.9695, predicted(3), 0.0001)

    assertEquals(0.36825, predicted(4), 0.0001)
    assertEquals(0.1892, predicted(5), 0.0001)
  }
}