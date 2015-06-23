package dk.gp.cogp.svi

import breeze.linalg.DenseMatrix
import dk.gp.cogp.svi.v.stochasticUpdateV
import dk.gp.cogp.svi.u.stochasticUpdateU
import dk.gp.cov.CovFunc
import dk.gp.cogp.svi.w.stochasticUpdateW
import dk.gp.cogp.svi.beta.stochasticUpdateBeta
import breeze.linalg.DenseVector

object stochasticUpdateLB {

  def apply(lbState: LBState, x: DenseMatrix[Double], y: DenseMatrix[Double], covFunc: CovFunc, covFuncParams:DenseVector[Double],l: Double): LBState = {

    val z = x //simplifying assumption
    val kXZ = covFunc.cov(x, z,covFuncParams)
    val kZZ = kXZ //simplifying assumption

    val newU = (0 until lbState.u.size).map { j => stochasticUpdateU(j, l, lbState, y, kXZ, kZZ) }.toArray

    val (newW, newWDelta) = stochasticUpdateW(lbState)
    val (newBeta, newBetaDelta) = stochasticUpdateBeta(lbState)

    val newV = (0 until lbState.v.size).map { i => stochasticUpdateV(i, l, lbState, y, kXZ, kZZ) }.toArray

    lbState.copy(u = newU, v = newV, beta = newBeta, betaDelta = newBetaDelta, w = newW, wDelta = newWDelta)
  }
}