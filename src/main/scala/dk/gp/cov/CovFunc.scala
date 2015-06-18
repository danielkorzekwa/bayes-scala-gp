package dk.gp.cov

import breeze.linalg.DenseMatrix

/**
 * Covariance function that measures similarity between two points in some input space.
 *
 */
trait CovFunc {

  def cov(x: DenseMatrix[Double]): DenseMatrix[Double] = {

    val covMatrix = DenseMatrix.zeros[Double](x.rows, x.rows)

    for (rowIndex <- 0 until x.rows) {

      val x1Val = x(rowIndex, ::).t.toArray

      for (colIndex <- 0 until x.rows) {

        val x2Val = x(colIndex, ::).t.toArray
        covMatrix(rowIndex, colIndex) = cov(x1Val, x2Val)
      }
    }

    covMatrix
  }

  def covNM(n: DenseMatrix[Double], m: DenseMatrix[Double]): DenseMatrix[Double] = {
    val covMatrix = DenseMatrix.zeros[Double](n.rows, m.rows)

    for (rowIndex <- 0 until n.rows) {

      val x1Val = n(rowIndex, ::).t.toArray

      for (colIndex <- 0 until m.rows) {
        val x2Val = m(colIndex, ::).t.toArray

        covMatrix(rowIndex, colIndex) = cov(x1Val, x2Val)
      }
    }

    covMatrix
  }

  /**
   * Returns similarity between two vectors.
   *
   * @param x1 [Dx1] vector
   * @param x2 [Dx1] vector
   */
  def cov(x1: Array[Double], x2: Array[Double]): Double
}