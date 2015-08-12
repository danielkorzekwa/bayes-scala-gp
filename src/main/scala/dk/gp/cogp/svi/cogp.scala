package dk.gp.cogp.svi

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.cov.CovFunc
import dk.gp.math.MultivariateGaussian
import dk.gp.math.MultivariateGaussian
import breeze.stats._
import breeze.linalg.inv
import breeze.linalg.sum
import breeze.linalg._
import dk.gp.cogp.CogpModel
import dk.gp.cogp.CogpGPVar

object cogp {

  def apply(x: DenseMatrix[Double], y: DenseMatrix[Double],
            covFuncG: Array[CovFunc], covFuncGParams: Array[DenseVector[Double]],
            covFuncH: Array[CovFunc], covFuncHParams: Array[DenseVector[Double]],
            l: Double, iterNum: Int): CogpModel = {

    val initialModel = CogpModel(x, y, covFuncG, covFuncGParams, covFuncH, covFuncHParams)

    val finalLBState = (0 until iterNum).foldLeft(initialModel) {

      case (currLBState, iter) => stochasticUpdateLB(currLBState, x, y, l)
    }
    finalLBState
  }

  // private def getInitialV()

  private def getInitialIndVarV(y: DenseVector[Double]): MultivariateGaussian = {
    val m = DenseVector.zeros[Double](y.size)

    val vInv = 0.1 * (1.0 / variance(y)) * DenseMatrix.eye[Double](y.size)
    val v = inv(vInv)
    MultivariateGaussian(m, v)
  }
}