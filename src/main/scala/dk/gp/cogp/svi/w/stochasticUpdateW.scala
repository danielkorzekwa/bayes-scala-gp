package dk.gp.cogp.svi.w

import breeze.linalg.DenseMatrix
import dk.gp.cogp.svi.LBState

object stochasticUpdateW {

  /**
   * Returns [new w, w delta]
   */
  def apply(lbState:LBState):(DenseMatrix[Double],DenseMatrix[Double]) = {
(lbState.w,lbState.wDelta)
  }
}