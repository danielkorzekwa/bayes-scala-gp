package dk.gp.gp

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
object gpPredict {

  /**
   * @param t Matrix of points for which p(y) is computed
   * @param f
   * @param x
   * @param covFunc
   * @param covFuncParams
   * @param mean
   *
   * @return Vector of p(y_i) for t points {i=1 to n}
   */
  def apply(t: DenseMatrix[Double], f: dk.gp.math.MultivariateGaussian, x: DenseMatrix[Double], covFunc: CovFunc, covFuncParams: DenseVector[Double], mean: Double = 0): DenseVector[dk.gp.math.MultivariateGaussian] = {
    
    val condGPFactory = ConditionalGPFactory(x,covFunc,covFuncParams,mean)

    val predicted = (0 until t.rows).par.map { i =>
      val tRow = t(i, ::).t

      val (a,b,v) = condGPFactory.create(tRow.toDenseMatrix)
      
      val fVariable = MultivariateGaussian(f.m, f.v)

      val yVariable = Gaussian(a, fVariable, b, v)
      val yPosterior = infer(yVariable)
      dk.gp.math.MultivariateGaussian(yPosterior.m, yPosterior.v)
    }.toArray

    DenseVector(predicted)
  }
}