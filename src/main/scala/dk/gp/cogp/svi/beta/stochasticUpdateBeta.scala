package dk.gp.cogp.svi.beta

import breeze.linalg.DenseVector
import dk.gp.cogp.CogpModel

object stochasticUpdateBeta {

  def apply(model: CogpModel): (DenseVector[Double], DenseVector[Double]) = {
    (model.beta, model.betaDelta)
  }
}