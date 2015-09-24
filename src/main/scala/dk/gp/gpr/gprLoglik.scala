package dk.gp.gpr

import scala.math.Pi
import scala.math.log

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.InjectNumericOps
import breeze.linalg.cholesky
import dk.gp.math.logdetchol

object gprLoglik {

  def apply(xMean: DenseVector[Double], kXX: DenseMatrix[Double], kXXInv: DenseMatrix[Double], y: DenseVector[Double]): Double = {
   
    val m = xMean
    
    val logDet = logdetchol(cholesky(kXX))
    
    val loglikValue = (-0.5 * (y - m).t * kXXInv * (y - m) - 0.5 * logDet - 0.5 * xMean.size.toDouble * log(2 * Pi))
    
    loglikValue(0)
  }
}