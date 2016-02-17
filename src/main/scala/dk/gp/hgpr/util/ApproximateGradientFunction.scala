package dk.gp.hgpr.util

import dk.gp.hgpr.HgprModel
import breeze.optimize.DiffFunction
import com.typesafe.scalalogging.slf4j.LazyLogging
import breeze.linalg.DenseVector
import breeze.linalg._

case class ApproximateGradientFunction (initialModel: HgprModel) extends DiffFunction[DenseVector[Double]] with LazyLogging {
  
   private val epsilon = 1E-5

  def calculate(x: DenseVector[Double]) = {

    try {
    
    val currCovFuncParams = DenseVector(x.toArray.dropRight(1))
    val currLikNoiseLogStdDev = x.toArray.last

    val currModel = initialModel.copy(covFuncParams = currCovFuncParams, likNoiseLogStdDev = currLikNoiseLogStdDev)

    val loglik =  -calcHgprLoglik(currModel) 
    
    val grad: DenseVector[Double] = DenseVector.zeros[Double](x.size)
    val xx = x.copy
    for ((k, v) <- x.iterator) {
      xx(k) += epsilon
      val gradModel = initialModel.copy(covFuncParams = DenseVector(xx.toArray.dropRight(1)), likNoiseLogStdDev = xx.toArray.last)
      val gradLoglik = -calcHgprLoglik(gradModel)
      grad(k) = (gradLoglik - loglik) / epsilon
      xx(k) -= epsilon
    }
    (loglik, grad)

      } catch {
      case e: NotConvergedException    => (Double.NaN, DenseVector.zeros[Double](x.size) * Double.NaN)
      case e: IllegalArgumentException => (Double.NaN, DenseVector.zeros[Double](x.size) * Double.NaN)
        case e: MatrixNotSymmetricException => (Double.NaN, DenseVector.zeros[Double](x.size) * Double.NaN)
    }
  }
}