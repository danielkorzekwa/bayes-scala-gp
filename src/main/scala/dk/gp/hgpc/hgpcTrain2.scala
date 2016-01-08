package dk.gp.hgpc

import breeze.linalg.NotConvergedException
import breeze.linalg.DenseVector
import breeze.optimize.LBFGS
import breeze.linalg._
import com.typesafe.scalalogging.slf4j._
import util._

/**
 * Hierarchical Gaussian Process classification. Multiple Gaussian Processes for n tasks with a single shared parent GP.
 */
object hgpcTrain2 extends LazyLogging {

  def apply(hgpcModel: HgpcModel, maxIter: Int = 100): HgpcModel = {

    val diffFunc = ApproximateGradientFunction(hgpcModel)

    val initialParams = DenseVector(hgpcModel.covFuncParams.toArray :+ hgpcModel.mean)
    val optimizer = new LBFGS[DenseVector[Double]](maxIter, m = 6, tolerance = 1.0E-6)
    val optIterations = optimizer.iterations(diffFunc, initialParams).map { state => println("iter=%d, loglik=%.4f, params=%s".format(state.iter, state.value, state.x)); state }.toList
    val newParams = optIterations.last.x

    val newCovFuncParams = DenseVector(newParams.toArray.dropRight(1))
    val newMean = newParams.toArray.last

    val trainedModel = hgpcModel.copy(covFuncParams = newCovFuncParams, mean = newMean)

    trainedModel

  }

}