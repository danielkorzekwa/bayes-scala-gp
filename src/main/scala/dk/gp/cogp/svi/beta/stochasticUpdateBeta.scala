package dk.gp.cogp.svi.beta

import breeze.linalg.DenseVector
import dk.gp.cogp.CogpModel
import dk.gp.cogp.svi.classicalMomentum
import breeze.linalg.DenseMatrix

object stochasticUpdateBeta {

  private val learningRate = 1e-5
  private val momentum = 0.9

  def apply(model: CogpModel, x: DenseMatrix[Double], y: DenseMatrix[Double]): (DenseVector[Double], DenseVector[Double]) = {

    val betaGrad = calcLBGradBeta(model, x, y)

    val (newBeta, newBetaDelta) = classicalMomentum(model.beta, model.betaDelta, learningRate, momentum, betaGrad)

    (newBeta, newBetaDelta)
  }

}