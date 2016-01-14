package dk.gp.mtgpc

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.cov.CovFunc
import breeze.optimize.LBFGS

/**
 * Multi Task Gaussian Process Classification parameters learning: It is almost like standard Gaussian Process Classification, but with hyper parameters shared between task Gaussian Processes.
 */
object mtgpcTrain {

  /**
   * @param model
   * @param maxIter
   *
   * @return Learned (covFuncParams,gpMean)
   */
  def apply(model: MtgpcModel, maxIter: Int = 100): MtgpcModel = {

    val initialParams = DenseVector(model.covFuncParams.toArray :+ model.gpMean)

    val diffFunction = MtGpcDiffFunction(model)

    val optimizer = new LBFGS[DenseVector[Double]](maxIter, m = 6, tolerance = 1.0E-6)
    val optIterations = optimizer.iterations(diffFunction, initialParams).map { state => println("iter=%d, loglik=%.4f, params=%s".format(state.iter, state.value, state.x)); state }.toList
    val newParams = optIterations.last.x

    val newCovFuncParams = DenseVector(newParams.toArray.dropRight(1))
    val newGpMean = newParams.toArray.last

    val trainedModel = model.copy(covFuncParams = newCovFuncParams, gpMean = newGpMean)

    trainedModel
  }

}