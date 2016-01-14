package dk.gp.gpc

import breeze.linalg.DenseVector
import breeze.optimize.DiffFunction
import breeze.optimize.ApproximateGradientFunction
import util._
import breeze.linalg._

case class GpcDiffFunction(initialGpcModel: GpcModel) extends DiffFunction[DenseVector[Double]] {

  val epsilon = 1E-5

  val g = calcLoglik(initialGpcModel)(_)

  def calculate(x: DenseVector[Double]): (Double, DenseVector[Double]) = {

    val currCovFuncParams = DenseVector(x.toArray.dropRight(1))
    val currMean = x.toArray.last
    val currModel = initialGpcModel.copy(covFuncParams = currCovFuncParams, gpMean = currMean)

    val loglik = try { -calcGPCLoglik(currModel) }
    catch {
      case e: NotConvergedException => Double.NaN
    }

    val grad: DenseVector[Double] = DenseVector.zeros[Double](x.size)
    val xx = x.copy
    for ((k, v) <- x.iterator) {
      xx(k) += epsilon
      val gradModel = initialGpcModel.copy(covFuncParams = DenseVector(xx.toArray.dropRight(1)), gpMean = xx.toArray.last)
      val graLoglik = -calcGPCLoglik(gradModel)
      grad(k) = (graLoglik - loglik) / epsilon
      xx(k) -= epsilon
    }
    (loglik, grad)

  }

  private def calcLoglik(gpcModel: GpcModel)(params: DenseVector[Double]): Double = {

    val currCovFuncParams = DenseVector(params.toArray.dropRight(1))

    val currMean = params.toArray.last

    val currModel = gpcModel.copy(covFuncParams = currCovFuncParams, gpMean = currMean)
    val loglik = try { calcGPCLoglik(currModel) }
    catch {
      case e: NotConvergedException => Double.NaN
    }
    -loglik
  }

}