package dk.gp.hgpc

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.cov.CovSEiso
import dk.gp.cov.CovFunc
import breeze.numerics._

case class TestCovFunc() extends CovFunc {

  private val covSEiso = CovSEiso()

  def cov(x1: DenseMatrix[Double], x2: DenseMatrix[Double], covFuncParams: DenseVector[Double]): DenseMatrix[Double] = {

    val (task1Vec, task2Vec) = if (x1(0, 0) == x2(0, 0)) (DenseVector.zeros[Double](x1.rows), DenseVector.zeros[Double](x2.rows))
    else (DenseVector.zeros[Double](x1.rows), DenseVector.ones[Double](x2.rows))

    val accountIdCov = covSEiso.cov(task1Vec.toDenseMatrix.t, task2Vec.toDenseMatrix.t, DenseVector(log(1e-3), log(1e-10)))

    val logSf = covFuncParams(0)
    val logEllx1 = covFuncParams(1)
    val logEllx2 = covFuncParams(2)

    val covX1 = covSEiso.cov(x1(::, 1 to 1), x2(::, 1 to 1), DenseVector(logSf, logEllx1))
    val covX2 = covSEiso.cov(x1(::, 2 to 2), x2(::, 2 to 2), DenseVector(log(1), logEllx2))

    val x1x2Cov = covX1 :* covX2 //this is equivalent to ARD iso kernel

    accountIdCov + x1x2Cov
  }

  def covD(x1: DenseMatrix[Double], x2: DenseMatrix[Double], covFuncParams: DenseVector[Double]): Array[DenseMatrix[Double]] =
    throw new UnsupportedOperationException("Not implemented yet")

}