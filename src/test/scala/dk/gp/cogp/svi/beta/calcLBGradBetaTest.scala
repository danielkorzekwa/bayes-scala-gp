package dk.gp.cogp.svi.beta

import org.junit._
import Assert._
import breeze.linalg._
import java.io.File
import dk.gp.cov.CovFunc
import dk.gp.cov.CovSEiso
import breeze.numerics.log
import dk.gp.cogp.CogpModel

class calcLBGradBetaTest {

  val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 4, ::)
  val x = data(::, 0).toDenseMatrix.t
  val y = data(::, 1 to 2)

  val covFuncG: Array[CovFunc] = Array(CovSEiso())
  val cofFuncGParams = Array(DenseVector(log(1), log(1)))

  val covFuncH: Array[CovFunc] = Array(CovSEiso(), CovSEiso())
  val covFuncHParams = Array(DenseVector(log(1), log(1e-10)), DenseVector(log(1), log(1e-10)))

  //@TODO the code for creating this CogpModel is placed many times in different test classes, put it in a single place
  val model = CogpModel(x, y, covFuncG, cofFuncGParams, covFuncH, covFuncHParams)

  @Test def test = {

    val model = CogpModel(x, y, covFuncG, cofFuncGParams, covFuncH, covFuncHParams)

    val betaGrad = calcLBGradBeta(model, x, y)
    assertEquals(-0.96381, betaGrad(0), 0.00001)
    assertEquals(-0.83919, betaGrad(1), 0.00001)

  }
}