package dk.gp.cov.utils

import org.junit._
import Assert._
import dk.gp.cov.CovSEiso
import scala.math._
import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix
import breeze.linalg.diag

class covDiagTest {

  val covFunc = CovSEiso()
  val covFuncParams = DenseVector(log(2), log(1))

  @Test def test = {

    val x = DenseMatrix.rand(5, 2)

    val cov = covFunc.cov(x, x, covFuncParams)
    val covDiagVal = covDiag(x, covFunc, covFuncParams)

    assertEquals(covDiagVal, diag(cov))
  }
}