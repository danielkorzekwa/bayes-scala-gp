package dk.gp.cogp.svi.w

import breeze.linalg.DenseMatrix
import dk.gp.cogp.CogpModel
import dk.gp.cogp.lb.LowerBound
import dk.gp.cogp.svi.classicalMomentum

object stochasticUpdateW {

  private val learningRate = 1e-5
  private val momentum = 0.9

  /**
   * Returns [new w, w delta]
   */
  def apply(lowerBound:LowerBound,model: CogpModel, x: DenseMatrix[Double], y: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {

    val wGrad = calcLBGradW(lowerBound,model, x, y)

    val (newW, newWDelta) = classicalMomentum(model.w, model.wDelta, learningRate, momentum, wGrad)

    (newW, newWDelta)
  }
}