package dk.gp.cogp.svi

import breeze.linalg.DenseMatrix
import dk.gp.cogp.lb.LowerBound
import dk.gp.cogp.lb.grad.calcLBGradW

object stochasticUpdateW {

  private val learningRate = 1e-5
  private val momentum = 0.9

  /**
   * Returns [new w, w delta]
   */
  def apply(lowerBound:LowerBound, y: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {

    val wGrad = calcLBGradW(lowerBound, y)

    val (newW, newWDelta) = classicalMomentum(lowerBound.model.w, lowerBound.model.wDelta, learningRate, momentum, wGrad)

    (newW, newWDelta)
  }
}