package dk.gp.util

import breeze.linalg.DenseMatrix

object filterRows {

  def apply(m: DenseMatrix[Double], colId: Int, f: Double => Boolean) = {
    val col = m(::, colId)
    val idx = col.findAll(f)
    m(idx, ::).toDenseMatrix
  }
}