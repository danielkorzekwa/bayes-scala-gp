package dk.gp.mtgp

import org.junit._
import Assert._
import breeze.linalg._
import java.io.File
import dk.gp.cov.CovSEiso
import breeze.numerics._

class learnMtGgHyperParamsTest {

  //[x,y]
  val data = csvread(new File("src/test/resources/gp/gpml_regression_data.csv"), skipLines = 1)
  val x = data(::, 0 to 0)
  val y = data(::, 1)

  val x1 = DenseMatrix.horzcat(DenseMatrix.zeros[Double](y.size, 1) + 1.0, x)
  val x2 = DenseMatrix.horzcat(DenseMatrix.zeros[Double](y.size, 1) + 1.0, x)

  val allX = DenseMatrix.vertcat(x1, x2)
  val allY = DenseVector.vertcat(y, y)

  val covFunc = CovSEiso()
  val covFuncParams = DenseVector(log(1d), log(1))
  val likNoiseLogStdDev = log(0.1)

  @Test def test = {

    val (learnedCovFuncParams, learnedLikNoiseLogStdDev) = learnMtGgHyperParams(allX, allY, covFunc, covFuncParams, likNoiseLogStdDev)

    assertEquals(0.66160, learnedCovFuncParams(0), 0.0001)
    assertEquals(-1.0825, learnedCovFuncParams(1), 0.0001)
    assertEquals(-2.1626, learnedLikNoiseLogStdDev, 0.0001)
  }
}