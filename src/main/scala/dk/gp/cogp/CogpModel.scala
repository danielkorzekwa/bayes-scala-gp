package dk.gp.cogp

import dk.gp.cov.CovFunc
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.math.MultivariateGaussian
import dk.gp.cov.utils.covDiag
import breeze.linalg.inv
import breeze.stats._
import breeze.linalg._
import dk.gp.math.invchol

case class CogpModel(g: Array[CogpGPVar], h: Array[CogpGPVar],
                     beta: DenseVector[Double], betaDelta: DenseVector[Double],
                     w: DenseMatrix[Double], wDelta: DenseMatrix[Double]) {

}

object CogpModel {

  def apply(x: DenseMatrix[Double], y: DenseMatrix[Double],
            covFuncG: Array[CovFunc], covFuncGParams: Array[DenseVector[Double]],
            covFuncH: Array[CovFunc], covFuncHParams: Array[DenseVector[Double]]): CogpModel = {

    //likelihood noise precision
    val beta = DenseVector(1d / 0.01, 1d / 0.01) // [P x 1]
    val betaDelta = DenseVector.zeros[Double](beta.size)

    //mixing weights
    val w = new DenseMatrix(2, 1, Array(1.0, 1)) // [P x Q]
    val wDelta = DenseMatrix.zeros[Double](w.rows, w.cols)

    val priorG = covFuncG.zip(covFuncGParams).map { case (covFunc, covFuncParams) => 
      CogpGPVar(x, getInitialIndVarV(mean(y(*, ::))), covFunc, covFuncParams,DenseVector.zeros[Double](covFuncParams.size)) }

    val priorH = covFuncH.zip(covFuncHParams).zipWithIndex.map { case ((covFunc, covFuncParams), i) => 
      CogpGPVar(x, getInitialIndVarV(y(::, i)), covFunc, covFuncParams,DenseVector.zeros[Double](covFuncParams.size)) }

    val lbState = CogpModel(priorG, priorH, beta, betaDelta, w, wDelta)

    lbState
  }

  private def getInitialIndVarV(y: DenseVector[Double]): MultivariateGaussian = {
    val m = DenseVector.zeros[Double](y.size)

    val vInv = 0.1 * (1.0 / variance(y)) * DenseMatrix.eye[Double](y.size)
   val v = invchol(cholesky(vInv).t)
    MultivariateGaussian(m, v)
  }
}