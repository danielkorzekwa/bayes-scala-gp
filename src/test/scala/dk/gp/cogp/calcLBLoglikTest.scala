package dk.gp.cogp

import org.junit._
import org.junit.Assert._
import breeze.linalg._
import java.io.File
import dk.gp.cov.CovFunc
import dk.gp.cov.CovSEiso
import breeze.numerics.log

class calcLBLoglikTest {

  val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 4, ::)
  val x = data(::, 0).toDenseMatrix.t
  val y = data(::, 1 to 2)

  val covFuncG: Array[CovFunc] = Array(CovSEiso())
  val cofFuncGParams = Array(DenseVector(log(1), log(1)))

  val covFuncH: Array[CovFunc] = Array(CovSEiso(), CovSEiso())
  val covFuncHParams = Array(DenseVector(log(1), log(1e-10)), DenseVector(log(1), log(1e-10)))

  @Test def test = {
    val model = CogpModel(x, y, covFuncG, cofFuncGParams, covFuncH, covFuncHParams)

    val loglik = calcLBLoglik(model, x, y)
    assertEquals(-121201.19002206407, loglik, 0.0000001)
  }
}