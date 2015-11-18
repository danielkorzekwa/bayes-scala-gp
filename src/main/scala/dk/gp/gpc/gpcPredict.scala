package dk.gp.gpc

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.bayes.dsl.variable.gaussian.multivariate.MultivariateGaussianFactor
import dk.bayes.dsl.variable.gaussian.multivariate.MultivariateGaussian
import dk.bayes.dsl.variable.categorical.MvnGaussianThreshold
import dk.bayes.dsl.infer
import breeze.linalg._
import dk.gp.math._
import breeze.numerics._
import dk.gp.gp._
import dk.bayes.math.gaussian.Gaussian
import dk.bayes.infer.epnaivebayes._
import dk.bayes.math.gaussian.canonical._

/**
 * Gaussian Process classification.
 */
object gpcPredict {

  /**
   * @param t Matrix of N test points [NxD], D - dimensionality of predictor vector
   * @param model
   *
   * @returns Predicted probabilities for class +1
   */
  def apply(t: DenseMatrix[Double], model: GpcModel): DenseVector[Double] = {

    val fPosterior = inferFPosterior(model)

    val predictedT = gpPredict(t, fPosterior, model.x, model.covFunc, model.covFuncParams, model.mean)

    val predictedProb = predictedT.map(predictedT => Gaussian.stdCdf(predictedT.m(0) / sqrt(1d + predictedT.v(0, 0))))

    predictedProb
  }

  private def inferFPosterior(model: GpcModel): dk.gp.math.MultivariateGaussian = {
    val covX = model.covFunc.cov(model.x, model.x, model.covFuncParams) + DenseMatrix.eye[Double](model.x.rows) * 1e-7
    val meanX = DenseVector.zeros[Double](model.x.rows) + model.mean

    val fVariable = MultivariateGaussian(meanX, covX)

    val yVariables = model.y.toArray.zipWithIndex.map {
      case (y, i) =>
        val isTrue = (y == 1)
        MvnGaussianThreshold(fVariable, i, v = 1d, exceeds = Some(isTrue)) //step function loglik with noise var = 1 is equivalent to probit likelihood
    }

    val factorGraph = EPNaiveBayesFactorGraph(fVariable, yVariables, true)
    factorGraph.calibrate(maxIter = 10, threshold = 1e-4)

    val fPosterior = factorGraph.getPosterior().asInstanceOf[DenseCanonicalGaussian]

    dk.gp.math.MultivariateGaussian(fPosterior.mean, fPosterior.variance)
  }

}