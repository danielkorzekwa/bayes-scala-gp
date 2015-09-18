package dk.gp.gpr

import org.junit._
import Assert._
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import scala.math._
import dk.gp.cov.CovSEiso

class predictTest {

  val covFuncParams = DenseVector(log(7.5120), log(2.1887))
  val covFunc = CovSEiso()
  val noiseLogStdDev = log(0.81075)

  @Test def test_1d_inputs = {

    val x = DenseVector(1d, 2, 3).toDenseMatrix.t
    val y = DenseVector(1d, 4, 9)
    val z = DenseVector(1d, 2, 3, 4, 50).toDenseMatrix.t

    val gpModel = GprModel(x, y, covFunc, covFuncParams, noiseLogStdDev)
    val prediction = predict(z, gpModel)
    val expected = new DenseMatrix(5, 2, Array(0.878, 4.407, 8.614, 10.975, 0.00001, 1.246, 1.123, 1.246, 6.063, 57.087))

    assertEquals(expected.map(v => "%.3f".format(v)).toString, prediction.map(v => "%.3f".format(v)).toString())
  }
}