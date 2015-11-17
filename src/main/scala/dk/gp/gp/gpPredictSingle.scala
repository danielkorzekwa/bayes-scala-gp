package dk.gp.gp

import breeze.linalg.{ * => * }
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.cholesky
import breeze.linalg.inv
import dk.bayes.dsl.infer
import dk.bayes.dsl.variable.Gaussian
import dk.bayes.dsl.variable.gaussian.multivariate.MultivariateGaussian
import dk.gp.cov.CovFunc
import dk.gp.math.invchol

/**
 * Returns p(y) = integral of p(f)*p(y|f)df
 *
 * p(f) - Gaussian process latent variable
 * p(y|f) - Gaussian process conditional latent variable
 */
object gpPredictSingle {

  /**
   * @param t A single N dim variable for which p(y) is computed
   * @param f
   * @param x
   * @param covFunc
   * @param covFuncParams
   * @param mean
   *
   * @return p(y)
   */
  def apply(t: DenseMatrix[Double], f: dk.gp.math.MultivariateGaussian, x: DenseMatrix[Double], covFunc: CovFunc, covFuncParams: DenseVector[Double], mean: Double = 0d): dk.gp.math.MultivariateGaussian = {

    val kXX = covFunc.cov(x, x, covFuncParams) + DenseMatrix.eye[Double](x.rows) * 1e-7
    val meanX = DenseVector.zeros[Double](x.rows) + mean

    val kTT = covFunc.cov(t, t, covFuncParams) + DenseMatrix.eye[Double](t.rows) * 1e-7
    val meanT = DenseVector.zeros[Double](t.rows) + mean

    val kTX = covFunc.cov(t, x, covFuncParams)

    val lXX = cholesky(kXX).t
    
    val A = kTX * invchol(lXX)
    val b = meanT - A * meanX
    val kTXInvLXX = kTX * inv(lXX)
    val v = kTT - kTXInvLXX * kTXInvLXX.t

    val fVariable = MultivariateGaussian(f.m, f.v)

    val yVariable = Gaussian(A, fVariable, b, v)
    val yPosterior = infer(yVariable)

    val predicted = dk.gp.math.MultivariateGaussian(yPosterior.m, yPosterior.v)
    predicted
  }
}