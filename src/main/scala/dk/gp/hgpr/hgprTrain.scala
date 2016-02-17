package dk.gp.hgpr

import breeze.linalg.DenseVector
import breeze.optimize.LBFGS
import util.calcHgprLoglik
import dk.gp.hgpr.util.ApproximateGradientFunction

/**
 * Hierarchical Gaussian Process regression. Multiple Gaussian Processes for n tasks with a single shared parent GP.
 */
object hgprTrain {

  def apply(model: HgprModel,maxIter:Int=100): HgprModel = {

 val diffFunc = ApproximateGradientFunction(model)
    
    val initialParams = DenseVector(model.covFuncParams.toArray :+ model.likNoiseLogStdDev)

    val optimizer = new LBFGS[DenseVector[Double]](maxIter, m = 6, tolerance = 1.0E-6)
    val optIterations = optimizer.iterations(diffFunc, initialParams).map { state => println("iter=%d, loglik=%.4f, params=%s".format(state.iter, state.value, state.x)); state }.toList
    val newParams = optIterations.last.x

    val newCovFuncParams = DenseVector(newParams.toArray.dropRight(1))
    val newNoiseLogStdDev = newParams.toArray.last

    val trainedModel = model.copy(covFuncParams = newCovFuncParams, likNoiseLogStdDev = newNoiseLogStdDev)

    trainedModel
  }

}