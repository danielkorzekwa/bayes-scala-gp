package dk.gp.cogp.svi.beta

import breeze.linalg.DenseVector
import dk.gp.cogp.CogpModel
import dk.gp.cogp.svi.classicalMomentum
import breeze.linalg.DenseMatrix
import dk.gp.cogp.lb.LowerBound

object stochasticUpdateBeta {

  private val learningRate = 1e-5
  private val momentum = 0.9

  def apply(lowerBound:LowerBound,model: CogpModel, x: DenseMatrix[Double], y: DenseMatrix[Double]): (DenseVector[Double], DenseVector[Double]) = {

    val betaGrad = calcLBGradBeta(lowerBound,model, x, y)

    val (newBeta, newBetaDelta) = classicalMomentum(model.beta, model.betaDelta, learningRate, momentum, betaGrad)

    (newBeta, newBetaDelta)
  }

}