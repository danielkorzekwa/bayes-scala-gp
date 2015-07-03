package dk.gp.cogp.svi

import breeze.linalg.DenseMatrix
import dk.gp.cogp.svi.v.stochasticUpdateV
import dk.gp.cogp.svi.u.stochasticUpdateU
import dk.gp.cov.CovFunc
import dk.gp.cogp.svi.w.stochasticUpdateW
import dk.gp.cogp.svi.beta.stochasticUpdateBeta
import breeze.linalg.DenseVector
import dk.gp.cov.utils.covDiag

object stochasticUpdateLB {

  def apply(lbState: LBState, x: DenseMatrix[Double], y: DenseMatrix[Double], covFunc: CovFunc, covFuncParams: DenseVector[Double], l: Double): LBState = {

    val z = x //simplifying assumption
    val kXZ = covFunc.cov(x, z, covFuncParams) 
    val kZZ = kXZ  + (DenseMatrix.eye[Double](x.size) * 1e-7)
    val kXXDiag = covDiag(x, covFunc, covFuncParams)

    val newU = (0 until lbState.u.size).map { j => stochasticUpdateU(j, l, lbState, y, kXZ, kZZ) }.toArray

    val (newW, newWDelta) = stochasticUpdateW(lbState, kXZ, kZZ, kXXDiag,y)
   
    val (newBeta, newBetaDelta) = stochasticUpdateBeta(lbState)

    val newV = (0 until lbState.v.size).map { i => stochasticUpdateV(i, l, lbState, y, kXZ, kZZ) }.toArray

    lbState.copy(u = newU, v = newV, beta = newBeta, betaDelta = newBetaDelta, w = newW, wDelta = newWDelta)
  }

  // private def
}