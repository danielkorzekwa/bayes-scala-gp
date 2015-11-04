package dk.gp.mtgp

import breeze.linalg.DenseMatrix
import dk.gp.cov.CovFunc
import breeze.linalg.DenseVector
import dk.gp.sgpr.SgprModel
import breeze.optimize.LBFGS

/**
 * Multi Task Gaussian Process Regression parameters learning: It is almost like standard Gaussian Process Regression, but with hyper parameters shared between task Gaussian Processes.
 */
object learnMtGgHyperParams {

  /**
   * @param x [taskId, feature1, feature2,...]
   * @param y
   * @param covFunc
   * @param initialCovFuncParams
   * @param initialLikNoiseLogStdDev
   *
   * @return Learned (covFuncParams,likNoiseLogStdDev)
   */
  def apply(x: DenseMatrix[Double], y: DenseVector[Double], covFunc: CovFunc, initialCovFuncParams: DenseVector[Double], initialLikNoiseLogStdDev: Double): (DenseVector[Double], Double) = {

    val initialParams = DenseVector(initialCovFuncParams.toArray :+ initialLikNoiseLogStdDev)
    val diffFunction = MtGpDiffFunction(x, y, covFunc,initialCovFuncParams,initialLikNoiseLogStdDev)

    val optimizer = new LBFGS[DenseVector[Double]](maxIter = 100, m = 6, tolerance = 1.0E-6)
    val optIterations = optimizer.iterations(diffFunction, initialParams).toList

    val newParams = optIterations.last.x

    val newCovFuncParams = DenseVector(newParams.toArray.dropRight(1))
    val newLogNoiseStdDev = newParams.toArray.last

    (newCovFuncParams, newLogNoiseStdDev)
  }
}