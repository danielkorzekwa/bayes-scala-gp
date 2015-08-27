package dk.gp.cogp.svi.hypcovh

import org.junit._
import Assert._
import breeze.linalg._
import java.io.File
import dk.gp.cov.CovFunc
import dk.gp.cov.CovSEiso
import breeze.numerics.log
import dk.gp.cogp.CogpModel

class calcLBGradHypCovHTest {

  val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 4, ::)
  val x = data(::, 0).toDenseMatrix.t
  val y = data(::, 1 to 2)

  val covFuncG: Array[CovFunc] = Array(CovSEiso())
  val cofFuncGParams = Array(DenseVector(log(1), log(1)))

  val covFuncH: Array[CovFunc] = Array(CovSEiso(), CovSEiso())
  val covFuncHParams = Array(DenseVector(log(1), log(1e-10)), DenseVector(log(1), log(1e-10)))

  val model = CogpModel(x, y, covFuncG, cofFuncGParams, covFuncH, covFuncHParams)

  @Test def test = {

    val covParamsGrad0 = calcLBGradHypCovH(i = 0, model, x, y)
    assertEquals(-3.79732, covParamsGrad0(0), 0.00001)
    assertEquals(0, covParamsGrad0(1), 0.00001)

    val covParamsGrad1 = calcLBGradHypCovH(i = 1, model, x, y)
    assertEquals(-4.01906, covParamsGrad1(0), 0.00001)
    assertEquals(0, covParamsGrad1(1), 0.00001)

  }

}