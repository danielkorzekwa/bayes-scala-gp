package dk.gp.cogp.svi.w

import breeze.linalg.DenseMatrix
import dk.gp.cogp.svi.LBState
import dk.gp.cogp.svi.classicalMomentum
import breeze.linalg.DenseVector

object stochasticUpdateW {

  private val learningRate = 1e-5
  private val momentum = 0.9

  /**
   * Returns [new w, w delta]
   */
  def apply(lbState: LBState,kXZ:DenseMatrix[Double],kZZ:DenseMatrix[Double],kXXDiag:DenseVector[Double],
      y:DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {

    val wGrad = calcLBGradW(lbState.w,lbState.beta,lbState.u,lbState.v,kXZ,kZZ,kXXDiag,y)
    
    val (newW, newWDelta) = classicalMomentum(lbState.w, lbState.wDelta, learningRate, momentum, wGrad)

    (newW, newWDelta)
  }
}