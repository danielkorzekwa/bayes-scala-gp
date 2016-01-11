package dk.gp.mtgpc

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.cov.CovFunc
import breeze.optimize.LBFGS

/**
 * Multi Task Gaussian Process Classification parameters learning: It is almost like standard Gaussian Process Classification, but with hyper parameters shared between task Gaussian Processes.
 */
object learnMtGgcHyperParams {

  /**
   * @param x [taskId, feature1, feature2,...]
   * @param y Vector of {0,1}
   * @param covFunc
   * @param initialCovFuncParams
   * @param gpMean The mean value of the Gaussian Process
   *
   * @return Learned (covFuncParams,gpMean)
   */
  def apply(x: DenseMatrix[Double], y: DenseVector[Double], covFunc: CovFunc, covFuncParams: DenseVector[Double], gpMean: Double): (DenseVector[Double], Double) = {
    val initialParams = DenseVector(covFuncParams.toArray :+ gpMean)

    val diffFunction = MtGpcDiffFunction(x, y, covFunc, covFuncParams, gpMean)

    val optimizer = new LBFGS[DenseVector[Double]](maxIter = 10, m = 6, tolerance = 1.0E-6)
    val optIterations = optimizer.iterations(diffFunction, initialParams).map { state => println("iter=%d, loglik=%.4f, params=%s".format(state.iter, state.value, state.x)); state }.toList
    val newParams = optIterations.last.x

    val newCovFuncParams = DenseVector(newParams.toArray.dropRight(1))
    val newGpMean = newParams.toArray.last

    (newCovFuncParams, newGpMean)
  }

}