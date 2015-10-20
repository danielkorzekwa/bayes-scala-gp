package dk.gp.sgpr

import breeze.linalg.DenseMatrix
import dk.gp.cov.utils.covDiagD
import breeze.linalg.DenseVector
import dk.gp.cov.CovFunc
import breeze.linalg.cholesky
import dk.gp.math.invchol
import breeze.numerics._
import dk.gp.cov.utils.covDiag

object calcLoglikWithD {

  /**
   * Returns Tuple3(
   * the value of lower bound,
   * derivatives of variational lower bound with respect to covariance hyper parameters,
   * derivatives of variational lower bound with respect to likelihood log noise std dev
   * )
   */
  def apply(x: DenseMatrix[Double], y: DenseVector[Double], u: DenseMatrix[Double], covFunc: CovFunc, covFuncParams: DenseVector[Double], logNoiseStdDev: Double): Tuple3[Double, Array[Double], Double] = {

    val likNoiseStdDev = exp(logNoiseStdDev)
    val likNoiseVar = likNoiseStdDev * likNoiseStdDev

    val kMM: DenseMatrix[Double] = covFunc.cov(u, u, covFuncParams) + DenseMatrix.eye[Double](u.rows) * 1e-7 //add some jitter
    val kMMinv = invchol(cholesky(kMM).t)
    val kMN: DenseMatrix[Double] = covFunc.cov(u, x, covFuncParams)
    val kNM = kMN.t
    val kNNdiag = covDiag(x, covFunc, covFuncParams) + 1e-7

    val kNNDiagDArray = covDiagD(x, covFunc, covFuncParams) //calcKnnDiagDArray(covFunc)
    val kMMdArray = covFunc.covD(u, u, covFuncParams)
    val kNMdArray = covFunc.covD(x, u, covFuncParams)

    val n = x.rows
    val m = u.rows

    calcLowerBoundWithD(kMM, kMMinv, kMMdArray, kMN, kNM, kNMdArray, kNNdiag, kNNDiagDArray, y, likNoiseVar, n, m)

  }
}