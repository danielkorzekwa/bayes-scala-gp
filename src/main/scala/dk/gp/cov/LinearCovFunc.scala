package dk.gp.cov

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.math.dotProdPairwise
import breeze.numerics._

case class LinearCovFunc() extends CovFunc {

  def cov(x1: DenseMatrix[Double], x2: DenseMatrix[Double], covFuncParams: DenseVector[Double]): DenseMatrix[Double] = {
    val logSf = covFuncParams(0)

    val dotProdMatrixqDistMatrix = dotProdPairwise(x1.t, x2.t)
    val covMatrix = exp(2 * logSf) * dotProdMatrixqDistMatrix
    covMatrix
  }

  def covD(x1: DenseMatrix[Double], x2: DenseMatrix[Double], covFuncParams: DenseVector[Double]): Array[DenseMatrix[Double]] = {
    ???
  }

}