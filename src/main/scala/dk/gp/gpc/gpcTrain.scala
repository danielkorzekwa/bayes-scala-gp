package dk.gp.gpc

import breeze.optimize.ApproximateGradientFunction
import breeze.linalg.DenseVector
import breeze.optimize.LBFGS
import dk.gp.gpc.util.calcGPCLoglik
import breeze.linalg._

object gpcTrain {

  def apply(gpcModel: GpcModel, maxIter: Int = 100): GpcModel = {

    val diffFunc = new GpcDiffFunction(gpcModel)

    val initialParams = DenseVector(gpcModel.covFuncParams.toArray :+ gpcModel.mean)

    val optimizer = new LBFGS[DenseVector[Double]](maxIter, m = 6, tolerance = 1.0E-6)
    val optIterations = optimizer.iterations(diffFunc, initialParams).toList
    val newParams = optIterations.last.x

    val newCovFuncParams = DenseVector(newParams.toArray.dropRight(1))
    val newMean = newParams.toArray.last

    val trainedModel = gpcModel.copy(covFuncParams = newCovFuncParams, mean = newMean)

    trainedModel
  }

}