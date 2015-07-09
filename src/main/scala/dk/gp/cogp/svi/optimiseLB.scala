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

object optimiseLB {

  def apply(x: DenseMatrix[Double], y: DenseMatrix[Double], covFunc: CovFunc, covFuncParams: DenseVector[Double], l: Double, iterNum: Int): LBState = {

    val initialLBState = getInitialLBState(x, y, covFunc, covFuncParams)

    val finalLBState = (0 until iterNum).foldLeft(initialLBState) {
      case (currLBState, iter) => stochasticUpdateLB(currLBState, x, y, covFunc, covFuncParams, l)
    }
    finalLBState
  }

  private def getInitialLBState(x: DenseMatrix[Double], y: DenseMatrix[Double], covFunc: CovFunc, covFuncParams: DenseVector[Double]): LBState = {
    //likelihood noise precision
    val beta = DenseVector(1d / 0.01, 1d / 0.02) // [P x 1]
    val betaDelta = DenseVector.zeros[Double](beta.size)

    //mixing weights
    val w = new DenseMatrix(2, 1, Array(1.1, -1.2)) // [P x Q]
    val wDelta = DenseMatrix.zeros[Double](w.rows, w.cols)

    val z = x //simplifying assumption
    val kZZ = covFunc.cov(x, z, covFuncParams)

    val priorU = Array(getInitialIndVarV(mean(y(*, ::))))

    val priorV = (0 until y.cols).map(i => getInitialIndVarV(y(::, i))).toArray

    val lbState = LBState(priorU, priorV, beta, betaDelta, w, wDelta)

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