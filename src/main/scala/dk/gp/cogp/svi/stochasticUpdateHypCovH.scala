package dk.gp.cogp.svi

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.cogp.lb.LowerBound
import dk.gp.cogp.lb.grad.calcLBGradHypCovH

object stochasticUpdateHypCovH {
  
  private val learningRate = 1e-5
  private val momentum = 0.9
  
  def apply(i: Int, lowerBound: LowerBound):(DenseVector[Double], DenseVector[Double]) = {
  val hypParamsD = calcLBGradHypCovH(i, lowerBound)

    val (newHypParams, newBetaDelta) = classicalMomentum(lowerBound.model.h(i).covFuncParams, lowerBound.model.h(i).covFuncParamsDelta, learningRate, momentum, hypParamsD)

    (newHypParams, newBetaDelta)
  }
}