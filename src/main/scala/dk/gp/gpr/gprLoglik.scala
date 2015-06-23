package dk.gp.gpr

import breeze.linalg.DenseVector
import breeze.linalg.logdet
import breeze.linalg.DenseMatrix
import scala.math._

object gprLoglik {

  def apply(xMean: DenseVector[Double], kXX: DenseMatrix[Double], kXXInv: DenseMatrix[Double], y: DenseVector[Double]): Double = {
   
    val m = xMean
    
    val logDet = logdet(kXX)._2
    
    val loglikValue = (-0.5 * (y - m).t * kXXInv * (y - m) - 0.5 * logDet - 0.5 * xMean.size.toDouble * log(2 * Pi))
    
    loglikValue(0)
  }
}