package dk.gp.sgpr

import breeze.linalg.DenseMatrix
import dk.gp.cov.CovSEiso
import breeze.linalg.DenseVector
import breeze.optimize.DiffFunction
import dk.gp.cov.utils.covDiagD
import dk.gp.cov.CovFunc

case class SparseGpDiffFunction(x: DenseMatrix[Double], y: DenseVector[Double], u: DenseMatrix[Double], covFunc: CovFunc) extends DiffFunction[DenseVector[Double]] {

  /**
   * @param x Logarithm of [signal standard deviation,length-scale,likelihood noise standard deviation]
   */
  def calculate(params: DenseVector[Double]): (Double, DenseVector[Double]) = {

    val covFuncParams = params(0 until params.size - 1)
    val logNoiseStdDev = params(params.size - 1)

    val (loglik, loglikDKernel, loglikDLikNoise) = calcLoglikWithD(x, y, u, covFunc, covFuncParams, logNoiseStdDev)
    val negativeD = DenseVector(loglikDKernel.map(x => -x) :+ (-loglikDLikNoise))
    (-loglik, negativeD)
  }

}