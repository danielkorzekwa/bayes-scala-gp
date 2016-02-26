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
import dk.bayes.math.gaussian.canonical._
import dk.gp.gpc.util.calcLoglikGivenLatentVar
import dk.bayes.dsl.epnaivebayes.EPNaiveBayesFactorGraph
import com.typesafe.scalalogging.slf4j.LazyLogging
import dk.gp.gpc.util.calibrateGpcFactorGraph
import dk.gp.gpc.util.GpcFactorGraph

/**
 * Gaussian Process classification.
 */
object gpcPredict extends LazyLogging {

  /**
   * @param t Matrix of N test points [NxD], D - dimensionality of predictor vector
   * @param model
   *
   * @returns Predicted probabilities for class +1
   */
  def apply(t: DenseMatrix[Double], model: GpcModel): DenseVector[Double] = {

    val now = System.currentTimeMillis()
    // logger.info("Calibrating factor graph...")
    val gpcFactorGraph = GpcFactorGraph(model)
    val (calib, iters) = calibrateGpcFactorGraph(gpcFactorGraph, maxIter = 10)
    //if (iters >= 10) logger.warn(s"Factor graph did not converge in less than 10 iterations")
    // logger.info("Calibrating factor graph...done: " + (System.currentTimeMillis() - now))

    val fPosterior = gpcFactorGraph.fVariable.get.asInstanceOf[DenseCanonicalGaussian]

    val predictedT = gpPredict(t, dk.bayes.math.gaussian.MultivariateGaussian(fPosterior.mean, fPosterior.variance), model.x, model.covFunc, model.covFuncParams, model.gpMean)

    val predictedProb = predictedT.map(predictedT => calcLoglikGivenLatentVar(predictedT.m(0), predictedT.v(0, 0), 1d))

    predictedProb
  }

}