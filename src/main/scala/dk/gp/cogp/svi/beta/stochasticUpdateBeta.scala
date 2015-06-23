package dk.gp.cogp.svi.beta

import dk.gp.cogp.svi.LBState
import breeze.linalg.DenseVector

object stochasticUpdateBeta {

  def apply(lbState: LBState): (DenseVector[Double], DenseVector[Double]) = {
    (lbState.beta, lbState.betaDelta)
  }
}