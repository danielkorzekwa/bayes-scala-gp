package dk.gp.sgpr

import breeze.linalg.DenseMatrix
import dk.gp.cov.CovSEiso
import breeze.linalg.DenseVector
import breeze.optimize.DiffFunction
import dk.gp.cov.utils.covDiagD

case class SparseGpDiffFunction(x: DenseMatrix[Double], y: DenseVector[Double], u: DenseMatrix[Double]) extends DiffFunction[DenseVector[Double]] {

  /**
   * @param x Logarithm of [signal standard deviation,length-scale,likelihood noise standard deviation]
   */
  def calculate(params: DenseVector[Double]): (Double, DenseVector[Double]) = {
    val (sf, ell, likStdDev) = (params(0), params(1), params(2))

    val covFunc = CovSEiso()
    val covFuncParams = DenseVector(sf, ell)
    val gp = GenericSparseGPR(x, y, u, covFunc, covFuncParams, likStdDev)

    val kNNDiagDArray = covDiagD(x, covFunc, covFuncParams) //calcKnnDiagDArray(covFunc)
    val kMMdArray = covFunc.covD(u, u, covFuncParams)
    val kNMdArray = covFunc.covD(x, u, covFuncParams)

    val (loglik, loglikDKernel, loglikDLikNoise) = gp.loglikWithD(kMMdArray, kNMdArray, kNNDiagDArray)
    val negativeD = DenseVector(loglikDKernel.map(x => -x) :+ (-loglikDLikNoise))
    (-loglik, negativeD)
  }

}