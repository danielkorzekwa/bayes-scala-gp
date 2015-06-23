package dk.gp.gpr

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.trace

object gprLoglikD {

  def apply(xMean: DenseVector[Double], kXXInv: DenseMatrix[Double], y: DenseVector[Double], covElemWiseD: Array[DenseMatrix[Double]]): DenseVector[Double] = {

    val m = xMean
    val a = kXXInv * (y - m)
    val aaMinuskXXInv = (a * a.t - kXXInv)
    
    val covDArray = covElemWiseD.map { covElemWiseD => 0.5 * trace(aaMinuskXXInv * covElemWiseD) }

    DenseVector(covDArray)
  }
}