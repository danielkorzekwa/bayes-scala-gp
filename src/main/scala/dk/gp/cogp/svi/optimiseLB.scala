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

object optimiseLB {

  def apply(x: DenseMatrix[Double], y: DenseMatrix[Double],
            covFuncG: Array[CovFunc], covFuncGParams: Array[DenseVector[Double]],
            covFuncH: Array[CovFunc], covFuncHParams: Array[DenseVector[Double]],
            l: Double, iterNum: Int): CogpModel = {

    val initialLBState = getInitialLBState(x, y, covFuncG, covFuncGParams, covFuncH, covFuncHParams)

    val finalLBState = (0 until iterNum).foldLeft(initialLBState) {

      case (currLBState, iter) => stochasticUpdateLB(currLBState, x, y, l)
    }
    finalLBState
  }

  private def getInitialLBState(x: DenseMatrix[Double], y: DenseMatrix[Double],
                                covFuncG: Array[CovFunc], covFuncGParams: Array[DenseVector[Double]],
                                covFuncH: Array[CovFunc], covFuncHParams: Array[DenseVector[Double]]): CogpModel = {
    //likelihood noise precision
    val beta = DenseVector(1d / 0.01, 1d / 0.01) // [P x 1]
    val betaDelta = DenseVector.zeros[Double](beta.size)

    //mixing weights
    val w = new DenseMatrix(2, 1, Array(1.0, 1)) // [P x Q]
    val wDelta = DenseMatrix.zeros[Double](w.rows, w.cols)

    val priorG = covFuncG.zip(covFuncGParams).map { case (covFunc, covFuncParams) => CogpGPVar(x, getInitialIndVarV(mean(y(*, ::))), covFunc, covFuncParams) }

    val priorH = covFuncH.zip(covFuncGParams).zipWithIndex.map { case ((covFunc, covFuncParams), i) => CogpGPVar(x, getInitialIndVarV(y(::, i)), covFunc, covFuncParams) }

    val lbState = CogpModel(priorG, priorH, beta, betaDelta, w, wDelta)

    lbState

  }

  // private def getInitialV()

  private def getInitialIndVarV(y: DenseVector[Double]): MultivariateGaussian = {
    val m = DenseVector.zeros[Double](y.size)

    val vInv = 0.1 * (1.0 / variance(y)) * DenseMatrix.eye[Double](y.size)
    val v = inv(vInv)
    MultivariateGaussian(m, v)
  }
}