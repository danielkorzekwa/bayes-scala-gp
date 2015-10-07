package dk.gp.math

import breeze.numerics._
import breeze.linalg._

object diagProd {

  /**
   * @param a [n x m]
   * @param b [n x m]
   *
   * @return diagonal vector [n] of a product a*b'
   */
  def apply(a: DenseMatrix[Double], b: DenseMatrix[Double]): DenseVector[Double] = sum(a :* b, Axis._1)
}