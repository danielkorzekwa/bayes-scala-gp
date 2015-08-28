package dk.gp.cogp.svi

import org.junit._
import org.junit.Assert._
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.cov.CovSEiso
import scala.math._
import dk.gp.cov.CovSEiso
import breeze.linalg._
import java.io.File
import dk.gp.cov.CovFunc
import dk.gp.cogp.calcLBLoglik

class optimiseLBTest {

  val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 4, ::)
  val x = data(::, 0).toDenseMatrix.t
  val y = data(::, 1 to 2)

  //val x: DenseMatrix[Double] = DenseVector.rangeD(-10, 10, 1).toDenseMatrix.t
  //val y = DenseVector.horzcat(DenseVector.zeros[Double](x.size) + 1d, DenseVector.zeros[Double](x.size) + 2d)

  val covFuncG: Array[CovFunc] = Array(CovSEiso())
  val covFuncGParams = Array(DenseVector(log(1), log(1)))

  val covFuncH: Array[CovFunc] = Array(CovSEiso(), CovSEiso())
  val covFuncHParams = Array(DenseVector(log(1), log(1e-10)), DenseVector(log(1), log(1e-10)))

  @Test def test = {

    val newModel = cogp(x, y, covFuncG, covFuncGParams, covFuncH, covFuncHParams, iterNum = 20)

    val loglik = calcLBLoglik(newModel, x, y)
    assertEquals(-109.61519, loglik, 0.00001)

  }
}