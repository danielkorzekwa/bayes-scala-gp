package dk.gp.cov

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.numerics._
import dk.gp.math.sqDist

case class CovNoise() extends CovFunc {

  def cov(x1: DenseMatrix[Double], x2: DenseMatrix[Double], covFuncParams: DenseVector[Double]): DenseMatrix[Double] = {

    val logSf = covFuncParams(0)

    val sqDistMatrix = sqDist(x1.t, x2.t).map { x => if (x > 1e-16) 0.0 else 1.0 } //@TODO use eye(n) for cov(x,x)
    val covMatrix = exp(2 * logSf) * sqDistMatrix
    covMatrix
  }

  def covD(x1: DenseMatrix[Double], x2: DenseMatrix[Double], covFuncParams: DenseVector[Double]): Array[DenseMatrix[Double]] = {

    val logSf = covFuncParams(0)

    val sqDistMatrix = sqDist(x1.t, x2.t).map { x => if (x > 1e-16) 0.0 else 1.0 } //@TODO use eye(n) for cov(x,x)
    val covMatrixDSf = 2 * exp(2 * logSf) * sqDistMatrix
    Array(covMatrixDSf)
  }

}