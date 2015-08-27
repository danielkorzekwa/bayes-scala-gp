package dk.gp.cogp.svi.hypcovh

import org.junit._
import Assert._
import breeze.linalg._
import java.io.File
import dk.gp.cov.CovFunc
import dk.gp.cov.CovSEiso
import breeze.numerics.log
import dk.gp.cogp.CogpModel
import dk.gp.cogp.calcLBLoglik

class stochasticUpdateHypCovHTest {

  val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 4, ::)
  val x = data(::, 0).toDenseMatrix.t
  val y = data(::, 1 to 2)

  val covFuncG: Array[CovFunc] = Array(CovSEiso())
  val cofFuncGParams = Array(DenseVector(log(1), log(1)))

  val covFuncH: Array[CovFunc] = Array(CovSEiso(), CovSEiso())
  val covFuncHParams = Array(DenseVector(log(1), log(1e-10)), DenseVector(log(1), log(1e-10)))

  val model = CogpModel(x, y, covFuncG, cofFuncGParams, covFuncH, covFuncHParams)

  @Test def test = {

    val model = CogpModel(x, y, covFuncG, cofFuncGParams, covFuncH, covFuncHParams)

    val (newHypParams0, newHypParamsDelta0) = stochasticUpdateHypCovH(i = 0, model, x, y)
    val newH0 = model.h(0).copy(covFuncParams = newHypParams0, covFuncParamsDelta = newHypParamsDelta0)

    val (newHypParams1, newHypParamsDelta1) = stochasticUpdateHypCovH(i = 1, model, x, y)
    val newH1 = model.h(1).copy(covFuncParams = newHypParams1, covFuncParamsDelta = newHypParamsDelta1)

    val newModel = model.copy(h = Array(newH0, newH1))

    val loglik = calcLBLoglik(newModel, x, y)
    assertEquals(-121201.189716, loglik, 0.00001)

  }

}