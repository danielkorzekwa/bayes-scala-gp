package dk.gp.cov.utils

import dk.gp.cov.CovFunc
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg._

object covDiag {

  def apply(x: DenseMatrix[Double], covFunc: CovFunc, covFuncParams: DenseVector[Double]): DenseVector[Double] = {

    val diag = x(*, ::).map(r => covFunc.cov(r.toDenseMatrix, r.toDenseMatrix, covFuncParams)(0, 0))
    diag
  }
}