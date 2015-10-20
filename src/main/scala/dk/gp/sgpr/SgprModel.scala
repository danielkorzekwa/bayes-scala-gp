package dk.gp.sgpr

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.cov.CovFunc
import dk.gp.math.invchol
import breeze.linalg.cholesky
import breeze.numerics._

case class SgprModel private (kMMinv: DenseMatrix[Double], sigma: DenseMatrix[Double], sigmaKmnyVal: DenseVector[Double], u: DenseMatrix[Double], covFunc: CovFunc, covFuncParams: DenseVector[Double], logNoiseStdDev: Double) {

}

object SgprModel {

  /**
   * Gaussian Process Regression. It uses Gaussian likelihood and zero mean functions.
   * Based on: Variational Learning of Inducing Variables in Sparse Gaussian Processes (http://jmlr.org/proceedings/papers/v5/titsias09a/titsias09a.pdf)
   *
   * @param x Inputs. [NxD] matrix, where N - number of training examples, D - dimensionality of input space
   * @param y Targets. [Nx1] matrix, where N - number of training examples
   * @param u Inducing points. [NxD] matrix, where N - number of inducing points, D - dimensionality of input space
   * @param covFunc Covariance function
   * @param noiseLogStdDev Log of noise standard deviation of Gaussian likelihood function
   *
   */
  def apply(x: DenseMatrix[Double], y: DenseVector[Double], u: DenseMatrix[Double], covFunc: CovFunc, covFuncParams: DenseVector[Double], logNoiseStdDev: Double): SgprModel = {

    val likNoiseStdDev = exp(logNoiseStdDev)

    val kMM: DenseMatrix[Double] = covFunc.cov(u, u, covFuncParams) + DenseMatrix.eye[Double](u.rows) * 1e-7 //add some jitter
    val kMN: DenseMatrix[Double] = covFunc.cov(u, x, covFuncParams)
    val kNM = kMN.t

    val sigma = invchol(cholesky(kMM + pow(likNoiseStdDev, -2) * kMN * kNM).t)
    val sigmaKmnyVal = sigma * kMN * y

    val kMMinv = invchol(cholesky(kMM).t)

    new SgprModel(kMMinv, sigma, sigmaKmnyVal, u, covFunc, covFuncParams, logNoiseStdDev)
  }
}