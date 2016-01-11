package dk.gp.mtgpc

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.cov.CovSEiso
import dk.gp.cov.CovFunc
import breeze.numerics._

case class TestMtGprCovFunc() extends CovFunc {

  val covSEiso = CovSEiso()

  def cov(x1: DenseMatrix[Double], x2: DenseMatrix[Double], covFuncParams: DenseVector[Double]): DenseMatrix[Double] = {

    val logSf = covFuncParams(0)
    val logEllx1 = covFuncParams(1)
    val covX1 = covSEiso.cov(x1(::, 1 to 1), x2(::, 1 to 1), DenseVector(logSf, logEllx1))

    covX1
  }

  def covD(x1: DenseMatrix[Double], x2: DenseMatrix[Double], covFuncParams: DenseVector[Double]): Array[DenseMatrix[Double]] = {
    val logSf = covFuncParams(0)
    val logEllx1 = covFuncParams(1)
    val covX1D = covSEiso.covD(x1(::, 1 to 1), x2(::, 1 to 1), DenseVector(logSf, logEllx1))

    covX1D
  }
}