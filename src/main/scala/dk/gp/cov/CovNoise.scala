package dk.gp.cov

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.numerics._

//@TODO refactor this and other cov functions
case class CovNoise() extends CovFunc {

  def cov(x1: DenseMatrix[Double], x2: DenseMatrix[Double], covFuncParams: DenseVector[Double]): DenseMatrix[Double] = {
    val logSf = covFuncParams(0)

    def op(v1: DenseVector[Double], v2: DenseVector[Double]): Double = cov(v1.toArray, v2.toArray, logSf)
    val covMatrix = covFunc(x1, x2, op)
    covMatrix
  }

  def cov(x1: Array[Double], x2: Array[Double], sf: Double): Double = if (distance(x1, x2) < 1e-16) exp(2 * sf) else 0

  private def distance(x1: Array[Double], x2: Array[Double]): Double = {

    var distance = 0d
    var i = 0
    while (i < x1.size) {
      distance += pow(x1(i) - x2(i), 2)
      i += 1
    }

    distance
  }

  private def covFunc(x1: DenseMatrix[Double], x2: DenseMatrix[Double], op: (DenseVector[Double], DenseVector[Double]) => Double): DenseMatrix[Double] = {
    val matrix = DenseMatrix.zeros[Double](x1.rows, x2.rows)

    for (rowIndex <- 0 until x1.rows) {

      val x1Val = x1(rowIndex, ::).t

      for (colIndex <- 0 until x2.rows) {
        val x2Val = x2(colIndex, ::).t

        matrix(rowIndex, colIndex) = op(x1Val, x2Val)
      }
    }

    matrix
  }

  def covD(x1: DenseMatrix[Double], x2: DenseMatrix[Double], covFuncParams: DenseVector[Double]): Array[DenseMatrix[Double]] = {

    val logSf = covFuncParams(0)

    def dfDsf(v1: DenseVector[Double], v2: DenseVector[Double]): Double = df_dSf(v1.toArray, v2.toArray, logSf)
    val covMatrixDSf = covFunc(x1, x2, dfDsf)

    Array(covMatrixDSf)
  }

  def df_dSf(x1: Array[Double], x2: Array[Double], logSf: Double): Double = {
    require(x1.size == x2.size, "Vectors x1 and x2 have different sizes")

    if (distance(x1, x2) < 1e-16) 2 * exp(2 * logSf) else 0
  }
}