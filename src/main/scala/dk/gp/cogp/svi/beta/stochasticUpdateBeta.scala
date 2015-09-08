package dk.gp.cogp.svi.beta

import breeze.linalg.DenseVector
import dk.gp.cogp.CogpModel
import dk.gp.cogp.svi.classicalMomentum
import breeze.linalg.DenseMatrix
import dk.gp.cogp.lb.LowerBound

object stochasticUpdateBeta {

  private val learningRate = 1e-5
  private val momentum = 0.9

  def apply(lowerBound:LowerBound, y: DenseMatrix[Double]): (DenseVector[Double], DenseVector[Double]) = {

    val betaGrad = calcLBGradBeta(lowerBound, y)

    val (newBeta, newBetaDelta) = classicalMomentum(lowerBound.model.beta, lowerBound.model.betaDelta, learningRate, momentum, betaGrad)

    (newBeta, newBetaDelta)
  }

}