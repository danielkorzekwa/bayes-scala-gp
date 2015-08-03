package dk.gp.cogp.svi.w

import breeze.linalg.DenseMatrix
import dk.gp.cogp.svi.classicalMomentum
import breeze.linalg.DenseVector
import dk.gp.cogp.CogpModel

object stochasticUpdateW {

  private val learningRate = 1e-5
  private val momentum = 0.9

  /**
   * Returns [new w, w delta]
   */
  def apply(model: CogpModel, x: DenseMatrix[Double], y: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {

    val wGrad = calcLBGradW(model, x, y)

    val (newW, newWDelta) = classicalMomentum(model.w, model.wDelta, learningRate, momentum, wGrad)

    (newW, newWDelta)
  }
}