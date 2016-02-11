package dk.gp.gpc

import breeze.optimize.ApproximateGradientFunction
import breeze.linalg.DenseVector
import breeze.optimize.LBFGS
import dk.gp.gpc.util.calcGPCLoglik
import breeze.linalg._
import util._
import breeze.optimize._

object gpcTrain {

  def apply(gpcModel: GpcModel, maxIter: Int = 100): GpcModel = {

    val diffFunc = GpcLowerboundDiffFunction(gpcModel)

    val initialParams = DenseVector(gpcModel.covFuncParams.toArray :+ gpcModel.gpMean)

    val optimizer = new LBFGS[DenseVector[Double]](maxIter, m = 6, tolerance = 1.0E-6)
    val optIterations = optimizer.iterations(diffFunc, initialParams).toList
    val newParams = optIterations.last.x

    val newCovFuncParams = DenseVector(newParams.toArray.dropRight(1))
    val newGPMean = newParams.toArray.last

    val trainedModel = gpcModel.copy(covFuncParams = newCovFuncParams, gpMean = newGPMean)

    trainedModel
  }

}