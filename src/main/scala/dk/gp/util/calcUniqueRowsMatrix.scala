package dk.gp.util

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector

object calcUniqueRowsMatrix {

  /**
   *  Returns new matrix with distinct rows only.
   */
  def apply(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    val uRows = (0 until m.rows).map(i => m(i, ::).t).distinct
    val uniqueRowsMat = DenseVector.horzcat(uRows: _*).t

    uniqueRowsMat
  }
}