package dk.gp.gpc

import breeze.optimize.ApproximateGradientFunction
import breeze.linalg.DenseVector
import breeze.optimize.LBFGS
import dk.gp.gpc.util.calcGPCLoglik
import breeze.linalg._

object gpcTrain {

  def apply(gpcModel: GpcModel, maxIter: Int = 100): GpcModel = {

    val g = calcLoglik(gpcModel)(_)
    val diffFunc = new ApproximateGradientFunction(g)

    val initialParams = DenseVector(gpcModel.covFuncParams.toArray :+ gpcModel.mean)

    val optimizer = new LBFGS[DenseVector[Double]](maxIter, m = 6, tolerance = 1.0E-6)
    val optIterations = optimizer.iterations(diffFunc, initialParams).toList
    val newParams = optIterations.last.x

    val newCovFuncParams = DenseVector(newParams.toArray.dropRight(1))
    val newMean = newParams.toArray.last

    val trainedModel = gpcModel.copy(covFuncParams = newCovFuncParams, mean = newMean)

    trainedModel
  }

  private def calcLoglik(gpcModel: GpcModel)(params: DenseVector[Double]): Double = {

    val currCovFuncParams = DenseVector(params.toArray.dropRight(1))

    val currMean = params.toArray.last

    val currModel = gpcModel.copy(covFuncParams = currCovFuncParams, mean = currMean)
    val loglik = try { calcGPCLoglik(currModel)}
    catch {
      case e :NotConvergedException => Double.NaN
    }
    -loglik
  }
}