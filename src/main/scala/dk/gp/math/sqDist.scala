package dk.gp.math

import breeze.linalg.DenseMatrix
import breeze.linalg._

/**
 * Computes matrix of square distances between corresponding rows of matrices x1 and x2.
 * Implemnentation based on sq_dist function from gpml library for Gaussian processes
 *
 * @param x1 [D x N]
 * @param x1 [D x M]
 * @returns [M x M]
 */
object sqDist {

  def apply(x1: DenseMatrix[Double], x2: DenseMatrix[Double]) = {

    val t1 = -2.0 * (x1.t * x2)

    val t2 = t1(*, ::) + sum(x2 :* x2, Axis._0).t

    t2(::, *) + sum(x1 :* x1, Axis._0).t

  }
}