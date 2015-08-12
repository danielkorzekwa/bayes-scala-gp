package dk.gp.cogp.svi.beta

import org.junit._
import Assert._
import breeze.linalg._
import java.io.File
import dk.gp.cov.CovFunc
import dk.gp.cov.CovSEiso
import breeze.numerics.log
import dk.gp.cogp.CogpModel
import dk.gp.cogp.calcLBLoglik
import dk.gp.cogp.svi.w.stochasticUpdateW

class stochasticUpdateBetaTest {

  val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 4, ::)
  val x = data(::, 0).toDenseMatrix.t
  val y = data(::, 1 to 2)

  val covFuncG: Array[CovFunc] = Array(CovSEiso())
  val cofFuncGParams = Array(DenseVector(log(1), log(1)))

  val covFuncH: Array[CovFunc] = Array(CovSEiso(), CovSEiso())
  val covFuncHParams = Array(DenseVector(log(1), log(1e-10)), DenseVector(log(1), log(1e-10)))

  val model = CogpModel(x, y, covFuncG, cofFuncGParams, covFuncH, covFuncHParams)

  @Test def test = {

    val (newBeta, newBetaDelta) = stochasticUpdateBeta(model, x, y)

    val newModel = model.copy(beta = newBeta)

    val loglik = calcLBLoglik(newModel, x, y)
    assertEquals(-121201.1900057, loglik, 0.000001)

  }
}