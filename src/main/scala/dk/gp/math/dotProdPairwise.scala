package dk.gp.math

import breeze.linalg.DenseMatrix
import breeze.linalg._
import breeze.generic.UFunc

/**
 * Computes matrix of pair-wise dot products between matrices x1 and x2.
 *
 * @param x1 [D x N]
 * @param x1 [D x M]
 * @return matrix of square distances [N x M]
 */
object dotProdPairwise extends UFunc {

  implicit object implDMDM extends Impl2[DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double]] {
    def apply(x1: DenseMatrix[Double], x2: DenseMatrix[Double]): DenseMatrix[Double] = x1.t * x2
  }
}
  
  