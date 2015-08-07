package dk.gp.cogp.svi.w

import org.junit._
import Assert._
import breeze.linalg._
import java.io.File
import dk.gp.cov.CovFunc
import dk.gp.cov.CovSEiso
import breeze.numerics.log
import dk.gp.cogp.CogpModel

class calcLBGradWTest {

  val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 4, ::)
  val x = data(::, 0).toDenseMatrix.t
  val y = data(::, 1 to 2)

  val covFuncG: Array[CovFunc] = Array(CovSEiso())
  val cofFuncGParams = Array(DenseVector(log(1), log(1)))

  val covFuncH: Array[CovFunc] = Array(CovSEiso(), CovSEiso())
  val covFuncHParams = Array(DenseVector(log(1), log(1e-10)), DenseVector(log(1), log(1e-10)))

  @Test def test = {

    val model = CogpModel(x, y, covFuncG, cofFuncGParams, covFuncH, covFuncHParams)

    val wGrad = calcLBGradW(model, x, y)
    assertEquals(-0.43602, wGrad(0, 0), 0.00001)
    assertEquals(-0.43602, wGrad(1, 0), 0.00001)
  }
}