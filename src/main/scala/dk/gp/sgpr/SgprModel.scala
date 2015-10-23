package dk.gp.sgpr

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.cov.CovFunc
import dk.gp.math.invchol
import breeze.linalg.cholesky
import breeze.numerics._
import dk.gp.math.inveig
import breeze.linalg.diag
import breeze.stats._
import breeze.linalg.inv

case class SgprModel private (kMMinv: DenseMatrix[Double], u: DenseMatrix[Double],
                              covFunc: CovFunc, covFuncParams: DenseVector[Double], logNoiseStdDev: Double,
                              yKnmInvLmInvLa: DenseVector[Double], //[u x 1]
                              invLm: DenseMatrix[Double], //[u x u] 
                              invLa: DenseMatrix[Double] // [u x u]
                              ) {
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
    val likNoiseVar = likNoiseStdDev * likNoiseStdDev

    val kMM: DenseMatrix[Double] = covFunc.cov(u, u, covFuncParams)
    val kNM: DenseMatrix[Double] = covFunc.cov(x, u, covFuncParams)

    val lm: DenseMatrix[Double] = cholesky(kMM + 1e-7 * DenseMatrix.eye[Double](kMM.rows)).t
    val kMMinv = invchol(lm)
    val invLm = inv(lm)
    val KnmInvLm = kNM * invLm
    val C = KnmInvLm.t * KnmInvLm

    val a = likNoiseVar * DenseMatrix.eye[Double](u.rows) + C
    val la = cholesky(a).t
    val invLa: DenseMatrix[Double] = inv(la)
    val yKnmInvLmInvLa: DenseVector[Double] = ((y.t * kNM * invLm) * invLa).t

    new SgprModel(kMMinv, u, covFunc, covFuncParams, logNoiseStdDev, yKnmInvLmInvLa, invLm, invLa)
  }
}