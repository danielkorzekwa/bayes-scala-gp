package dk.gp.mtgpr

import breeze.linalg.DenseVector
import breeze.optimize.LBFGS

/**
 * Multi Task Gaussian Process Regression parameters learning: It is almost like standard Gaussian Process Regression, but with hyper parameters shared between task Gaussian Processes.
 */
object mtgprTrain {

  /**
   *
   * @return Learned model
   */
  def apply(model: MtGprModel, tolerance: Double = 1.0E-6): MtGprModel = {

    val initialParams = DenseVector(model.covFuncParams.toArray :+ model.likNoiseLogStdDev)
    val diffFunction = MtGpDiffFunction(model)

    val optimizer = new LBFGS[DenseVector[Double]](maxIter = 100, m = 6, tolerance)
    val optIterations = optimizer.iterations(diffFunction, initialParams).map { state => println("iter=%d, loglik=%.4f, params=%s".format(state.iter, state.value, state.x)); state }.toList

    val newParams = optIterations.last.x

    val newCovFuncParams = DenseVector(newParams.toArray.dropRight(1))
    val newLogNoiseStdDev = newParams.toArray.last

    val trainedModel = model.copy(covFuncParams = newCovFuncParams, likNoiseLogStdDev = newLogNoiseStdDev)
    trainedModel
  }

}