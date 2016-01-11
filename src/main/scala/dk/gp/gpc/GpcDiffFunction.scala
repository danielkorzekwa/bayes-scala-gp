package dk.gp.gpc

import breeze.linalg.DenseVector
import breeze.optimize.DiffFunction
import breeze.optimize.ApproximateGradientFunction
import util._
import breeze.linalg._

case class GpcDiffFunction(initialGpcModel: GpcModel) extends DiffFunction[DenseVector[Double]] {

  val g = calcLoglik(initialGpcModel)(_)
  val approxDiffFunc = new ApproximateGradientFunction(g)

  def calculate(params: DenseVector[Double]): (Double, DenseVector[Double]) = {
    approxDiffFunc.calculate(params)
  }

  private def calcLoglik(gpcModel: GpcModel)(params: DenseVector[Double]): Double = {

    val currCovFuncParams = DenseVector(params.toArray.dropRight(1))

    val currMean = params.toArray.last

    val currModel = gpcModel.copy(covFuncParams = currCovFuncParams, mean = currMean)
    val loglik = try { calcGPCLoglik(currModel) }
    catch {
      case e: NotConvergedException => Double.NaN
    }
    -loglik
  }

}