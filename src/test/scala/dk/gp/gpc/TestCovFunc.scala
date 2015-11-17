package dk.gp.gpc

import dk.gp.cov.CovFunc
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.cov.CovSEiso
import breeze.numerics._

case class TestCovFunc() extends CovFunc {

  val covSEiso = CovSEiso()

  def cov(x1: DenseMatrix[Double], x2: DenseMatrix[Double], covFuncParams: DenseVector[Double]): DenseMatrix[Double] = {

    val logSf = covFuncParams(0)
    val logEllx1 = covFuncParams(1)
    val logEllx2 = covFuncParams(2)

    val covX1 = covSEiso.cov(x1(::, 0 to 0), x2(::, 0 to 0), DenseVector(logSf, logEllx1))
    val covX2 = covSEiso.cov(x1(::, 1 to 1), x2(::, 1 to 1), DenseVector(log(1), logEllx2))

    covX1 :* covX2 //this is equivalent to ARD iso kernel
  }

  def covD(x1: DenseMatrix[Double], x2: DenseMatrix[Double], covFuncParams: DenseVector[Double]): Array[DenseMatrix[Double]] = {
    throw new UnsupportedOperationException("Not implemented yet")
  }
}