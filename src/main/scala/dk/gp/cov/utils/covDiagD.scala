package dk.gp.cov.utils

import dk.gp.cov.CovFunc
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg._

object covDiagD {

  def apply(x: DenseMatrix[Double], covFunc: CovFunc, covFuncParams: DenseVector[Double]): Array[DenseVector[Double]] = {

    val covDArray = Array.fill(covFuncParams.size)(DenseVector.zeros[Double](x.rows))

    for (i <- 0 until x.rows) {

      val covD = covFunc.covD(x(i, ::).t.toDenseMatrix, covFuncParams)
      covD.zipWithIndex.foreach { case (covD, index) => covDArray(index)(i) = covD(0, 0) }
    }

    covDArray
  }
}