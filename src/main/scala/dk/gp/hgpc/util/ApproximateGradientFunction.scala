package dk.gp.hgpc.util

import breeze.optimize.DiffFunction
import breeze.linalg.DenseVector
import dk.gp.hgpc.HgpcModel
import com.typesafe.scalalogging.slf4j._
import breeze.linalg.NotConvergedException
import breeze.linalg._

case class ApproximateGradientFunction(initialModel: HgpcModel) extends DiffFunction[DenseVector[Double]] with LazyLogging {

  private val epsilon = 1E-5

  def calculate(x: DenseVector[Double]) = {

    val currCovFuncParams = DenseVector(x.toArray.dropRight(1))
    val currMean = x.toArray.last

    val currModel = initialModel.copy(covFuncParams = currCovFuncParams, mean = currMean)

    val loglik = try { -calcHGPCLoglik(currModel) }
    catch {
      case e: NotConvergedException => Double.NaN
      case e: MatrixNotSymmetricException => {
        logger.warn(e.getLocalizedMessage)
        Double.NaN
      }
    }

    val grad: DenseVector[Double] = DenseVector.zeros[Double](x.size)
    val xx = x.copy
    for ((k, v) <- x.iterator) {
      xx(k) += epsilon
      val gradModel = initialModel.copy(covFuncParams = DenseVector(xx.toArray.dropRight(1)), mean = xx.toArray.last)
      val gradLoglik = -calcHGPCLoglik(gradModel)
      grad(k) = (gradLoglik - loglik) / epsilon
      xx(k) -= epsilon
    }
    (loglik, grad)

  }

}