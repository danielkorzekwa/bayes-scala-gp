package dk.gp.gpr

import org.junit._
import Assert._
import breeze.linalg._
import java.io.File
import scala.math._
import dk.gp.cov.CovSEiso

class gprTest {

  //[x,y]
  val data = csvread(new File("src/test/resources/gp/gpml_regression_data.csv"), skipLines = 1)

  val x = data(::, 0 to 0)
  val y = data(::, 1)

  val covFunc = CovSEiso()
  val covFuncParams = DenseVector(log(1d), log(1))
  val noiseLogStdDev = log(0.1)

  @Test def test = {

    val gpModel = gpr(x, y, covFunc, covFuncParams, noiseLogStdDev)

    assertEquals(0.68594, gpModel.covFuncParams(0), 0.0001)
    assertEquals(-0.99340, gpModel.covFuncParams(1), 0.0001)
    assertEquals(-1.9025, gpModel.noiseLogStdDev, 0.0001)

  }
}