package dk.gp.cov

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector

/**
 * Covariance function that measures similarity between two points in some input space.
 *
 */
trait CovFunc {

  def cov(x1:DenseMatrix[Double],x2:DenseMatrix[Double],covFuncParams:DenseVector[Double]):DenseMatrix[Double]
  
  def covD(x:DenseMatrix[Double],covFuncParams:DenseVector[Double]):Array[DenseMatrix[Double]]
 
}