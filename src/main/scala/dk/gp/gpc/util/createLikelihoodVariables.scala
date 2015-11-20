package dk.gp.gpc.util

import dk.bayes.dsl.variable.gaussian.multivariate.MultivariateGaussian
import dk.bayes.dsl.variable.categorical.MvnGaussianThreshold
import breeze.linalg.DenseVector

object createLikelihoodVariables {

  /**
   * @param prior
   * @param y Vector of {-1,1}
   */
  def apply(prior: MultivariateGaussian, y: DenseVector[Double]): Array[MvnGaussianThreshold] = {

    val yVariables = y.toArray.zipWithIndex.map {
      case (y, i) =>
        val isTrue = (y == 1)
        MvnGaussianThreshold(prior, i, v = 1d, exceeds = Some(isTrue)) //step function loglik with noise var = 1 is equivalent to probit likelihood
    }

    yVariables
  }
}