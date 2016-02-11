package dk.gp.gpc.util

import breeze.linalg.DenseVector
import breeze.optimize.DiffFunction
import breeze.optimize.ApproximateGradientFunction
import util._
import breeze.linalg._
import dk.gp.gpc.GpcModel

case class GpcApproxDiffFunction(initialGpcModel: GpcModel) extends DiffFunction[DenseVector[Double]] {

  val epsilon = 1E-5

  def calculate(x: DenseVector[Double]): (Double, DenseVector[Double]) = {
    try {
      val currCovFuncParams = DenseVector(x.toArray.dropRight(1))
      val currMean = x.toArray.last
      val currModel = initialGpcModel.copy(covFuncParams = currCovFuncParams, gpMean = currMean)

      val loglik = -calcGPCLoglik(currModel)

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

    } catch {
      case e: NotConvergedException    => (Double.NaN, DenseVector.zeros[Double](x.size) * Double.NaN)
      case e: IllegalArgumentException => e.printStackTrace(); System.exit(-1); (Double.NaN, DenseVector.zeros[Double](x.size) * Double.NaN)
    }
  }

}