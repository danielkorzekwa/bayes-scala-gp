package dk.gp.gp

import dk.bayes.dsl.infer
import dk.bayes.dsl.variable.gaussian.multivariate.MultivariateGaussian
import dk.bayes.dsl.variable.Gaussian
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.cov.CovFunc
import breeze.linalg.cholesky

/**
 * Returns p(y) = integral of p(f)*p(y|f)df
 *
 * p(f) - Gaussian process latent variable
 * p(y|f) - Gaussian process conditional latent variable
 *
 *
 */
case class GPPredictSingle(f: dk.bayes.math.gaussian.MultivariateGaussian, x: DenseMatrix[Double], covFunc: CovFunc, covFuncParams: DenseVector[Double], mean: Double = 0d) {

  private val condGPFactory = ConditionalGPFactory(x, covFunc, covFuncParams, mean)

  private val fvchol = cholesky(f.v)
  
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
  def predictSingle(t: DenseMatrix[Double]): dk.bayes.math.gaussian.MultivariateGaussian = {
    val (a, b, v) = condGPFactory.create(t)

    val skillMean = a * f.m + b
    val al = a*fvchol
    val skillVar = v + al*al.t
     
    val predicted = dk.bayes.math.gaussian.MultivariateGaussian(skillMean, skillVar)
    predicted
  }
}