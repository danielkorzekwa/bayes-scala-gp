package dk.gp.sgpr

import dk.gp.cov.CovFunc
import breeze.numerics._
import breeze.linalg._
import dk.gp.cov.utils.covDiag
import dk.gp.math.invchol
import dk.gp.math.UnivariateGaussian
import dk.gp.math.UnivariateGaussian

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
case class GenericSparseGPR(x: DenseMatrix[Double], y: DenseVector[Double], u: DenseMatrix[Double], covFunc: CovFunc, covFuncParams: DenseVector[Double], noiseLogStdDev: Double) extends SparseGPR {

  private val likNoiseStdDev = exp(noiseLogStdDev)
  private val likNoiseVar = likNoiseStdDev * likNoiseStdDev

  private val kMM: DenseMatrix[Double] = covFunc.cov(u, u, covFuncParams) + DenseMatrix.eye[Double](u.rows) * 1e-7 //add some jitter
  private val kMN: DenseMatrix[Double] = covFunc.cov(u, x, covFuncParams)
  private val kNM = kMN.t
  private val kNNdiag = covDiag(x, covFunc, covFuncParams) + 1e-7
  private val sigma = invchol(cholesky(kMM + pow(likNoiseStdDev, -2) * kMN * kNM).t)
  private val kMMinv = invchol(cholesky(kMM).t)
  private val yValue: DenseVector[Double] = y
   private val sigmaKmnyVal = sigma * kMN * yValue

  def predict(z: DenseMatrix[Double]): DenseVector[UnivariateGaussian] = {

    val predicted = z(*, ::).map { z =>

      val kZZ: DenseMatrix[Double] = covFunc.cov(z.toDenseMatrix, z.toDenseMatrix, covFuncParams)
      val kZU: DenseMatrix[Double] = covFunc.cov(z.toDenseMatrix, u, covFuncParams)
      val kUZ = kZU.t

      val predMean = pow(likNoiseStdDev, -2) * kZU * sigmaKmnyVal
      val predVariance = kZZ - kZU * kMMinv * kUZ + kZU * sigma * kUZ
      UnivariateGaussian(predMean(0), predVariance(0, 0))
    }

    //    val kZZ: DenseMatrix[Double] = covFunc.cov(z, z, covFuncParams)
    //    val kZU: DenseMatrix[Double] = covFunc.cov(z, u, covFuncParams)
    //    val kUZ = kZU.t
    //
    //    //@TODO use Cholesky Factorization instead of a direct inverse
    //    val predMean = pow(likNoiseStdDev, -2) * kZU * sigma * kMN * yValue
    //    val predVariance = kZZ - kZU * kMMinv * kUZ + kZU * sigma * kUZ
    //
    //    DenseVector.horzcat(predMean, diag(predVariance))

    predicted
  }

  /**
   * Returns Tuple3(
   * the value of lower bound,
   * derivatives of variational lower bound with respect to covariance hyper parameters,
   * derivatives of variational lower bound with respect to likelihood log noise std dev
   * )
   */
  def loglikWithD(kMMdArray: Array[DenseMatrix[Double]], kNMdArray: Array[DenseMatrix[Double]], kNNDiagDArray: Array[DenseVector[Double]]): Tuple3[Double, Array[Double], Double] = {

    val n = x.rows
    val m = u.rows

    calcLowerBoundWithD(kMM, kMMinv, kMMdArray, kMN, kNM, kNMdArray, kNNdiag, kNNDiagDArray, y, likNoiseVar, n, m)
  }

}