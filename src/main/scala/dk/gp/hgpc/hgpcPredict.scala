package dk.gp.hgpc

import com.typesafe.scalalogging.slf4j.LazyLogging

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian
import dk.gp.gp.ConditionalGPFactory
import dk.gp.gp.GPPredictSingle
import dk.gp.gpc.util.calcLoglikGivenLatentVar
import dk.gp.hgpc.util.HgpcFactorGraph
import dk.gp.hgpc.util.calibrateHgpcFactorGraph
import dk.gp.math.MultivariateGaussian

/**
 * Hierarchical Gaussian Process classification. Multiple Gaussian Processes for n tasks with a single shared parent GP.
 */
object hgpcPredict extends LazyLogging {

  case class TaskPosterior(x: DenseMatrix[Double], xPosterior: DenseCanonicalGaussian)

  /**
   * Returns vector of probabilities for the class 1
   */
  def apply(t: DenseMatrix[Double], model: HgpcModel): DenseVector[Double] = hgpcPredictDetailed(t, model)(::, 2)

}