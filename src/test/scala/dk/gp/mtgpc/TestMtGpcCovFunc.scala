package dk.gp.mtgpc

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.cov.CovSEiso
import dk.gp.cov.CovFunc
import breeze.numerics._

case class TestMtGpcCovFunc() extends CovFunc {

  val covSEiso = CovSEiso()

  def cov(x1: DenseMatrix[Double], x2: DenseMatrix[Double], covFuncParams: DenseVector[Double]): DenseMatrix[Double] = {

    val logSf = covFuncParams(0)
    val logEllx1 = covFuncParams(1)
    val logEllx2 = covFuncParams(2)

    val covX1 = covSEiso.cov(x1(::, 1 to 1), x2(::, 1 to 1), DenseVector(logSf, logEllx1))
    val covX2 = covSEiso.cov(x1(::, 2 to 2), x2(::, 2 to 2), DenseVector(log(1), logEllx2))

    covX1 :* covX2 //this is equivalent to ARD iso kernel
  }

  def covD(x1: DenseMatrix[Double], x2: DenseMatrix[Double], covFuncParams: DenseVector[Double]): Array[DenseMatrix[Double]] = {
    val logSf = covFuncParams(0)
    val logEllx1 = covFuncParams(1)
    val logEllx2 = covFuncParams(2)

    val covX1 = covSEiso.cov(x1(::, 1 to 1), x2(::, 1 to 1), DenseVector(logSf, logEllx1))
    val covX2 = covSEiso.cov(x1(::, 2 to 2), x2(::, 2 to 2), DenseVector(log(1), logEllx2))

    val covDX1 = covSEiso.covD(x1(::, 1 to 1), x2(::, 1 to 1), DenseVector(logSf, logEllx1))
    val covDX2 = covSEiso.covD(x1(::, 2 to 2), x2(::, 2 to 2), DenseVector(log(1), logEllx2))

    val covDLogSf = covDX1(0) :* covX2
    val covDlogEllx1 = covDX1(1) :* covX2
    val covDlogEllx2 = covX1 :* covDX2(1)

    Array(covDLogSf, covDlogEllx1, covDlogEllx2)
  }
}