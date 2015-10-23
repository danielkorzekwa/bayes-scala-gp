package dk.gp.sgpr.lb

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.cov.CovFunc
import breeze.numerics._
import dk.gp.sgpr.lb.SgprLowerBound
import dk.gp.sgpr.lb.calcLowerBoundKLikNoise
import dk.gp.sgpr.lb.calcLowerBoundDKernel
import dk.gp.sgpr.lb.calcLowerBound

object calcLoglikWithD {

  /**
   * Returns Tuple3(
   * the value of lower bound,
   * derivatives of variational lower bound with respect to covariance hyper parameters,
   * derivatives of variational lower bound with respect to likelihood log noise std dev
   * )
   */
  def apply(x: DenseMatrix[Double], y: DenseVector[Double], u: DenseMatrix[Double], covFunc: CovFunc, covFuncParams: DenseVector[Double], logNoiseStdDev: Double): Tuple3[Double, Array[Double], Double] = {

    val lb = SgprLowerBound(x, y, u, covFunc, covFuncParams, logNoiseStdDev)

    val lBTerms: Array[Double] = calcLowerBound(lb)

    val lowerBoundVal = lBTerms.sum
    val lowerBoundDKernel = calcLowerBoundDKernel(lb)
    val lowerBoundDLikNoise = calcLowerBoundKLikNoise(lb, lBTerms(2), lBTerms(3), lBTerms(4))

    (lowerBoundVal, lowerBoundDKernel, lowerBoundDLikNoise)

  }
}