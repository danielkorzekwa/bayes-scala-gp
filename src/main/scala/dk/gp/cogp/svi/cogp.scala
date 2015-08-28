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
            iterNum: Int): CogpModel = {

    val initialModel = CogpModel(x, y, covFuncG, covFuncGParams, covFuncH, covFuncHParams)

    val finalLBState = (0 until iterNum).foldLeft(initialModel) {

      case (currLBState, iter) => stochasticUpdateLB(currLBState, x, y)
    }
    finalLBState
  }

}