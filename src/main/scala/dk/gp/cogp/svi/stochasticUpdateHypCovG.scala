package dk.gp.cogp.svi

import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix
import breeze.generic.UFunc
import dk.gp.cogp.lb.LowerBound
import dk.gp.cogp.lb.grad.calcLBGradHypCovG

object stochasticUpdateHypCovG {

  private val learningRate = 1e-5
  private val momentum = 0.9

  def apply(j: Int, lowerBound:LowerBound, y: DenseMatrix[Double]): (DenseVector[Double], DenseVector[Double]) = {

    val hypParamsD = calcLBGradHypCovG(j, lowerBound, y)

    val (newHypParams, newHypParamsDelta) = classicalMomentum(lowerBound.model.g(j).covFuncParams, lowerBound.model.g(j).covFuncParamsDelta, learningRate, momentum, hypParamsD)

    (newHypParams, newHypParamsDelta)
  }

}