package dk.gp.util

import breeze.linalg._
import breeze.linalg._
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.stats._
import breeze.stats.mean.reduce_Double

/**
 * Creates matrix with n vectors placed on the diagonal of the matrix. Off diagonal of the matrix equals to the means of the input vectors. Example explains it the best:
 *
 * input: [1 2],[3,4]
 *
 * output:
 *
 * [1   3.5
 *  2   3.5
 *  1.5 3
 *  1.5 4]
 *
 * It is used for computing inducing points matrix for additive kernel
 *
 * @param dimVectors
 *
 * @return matrix
 */
object calcInducingPointsMatrix {

  def apply(dimVectors: Array[DenseVector[Double]]): DenseMatrix[Double] = {

    val defaultRow = DenseVector(dimVectors.map(v => mean(v)))

    val uMatrices = dimVectors.zipWithIndex.map {
      case (vector, i) =>

        val zeroMatrix = DenseMatrix.zeros[Double](vector.size, dimVectors.size)
        val uMatrix = zeroMatrix(*, ::).map(r => defaultRow)
        uMatrix(::, i) := vector
        uMatrix
    }

    val u = DenseMatrix.vertcat(uMatrices: _*)
    val uniqueU = calcUniqueRowsMatrix(u)
    uniqueU
  }

}