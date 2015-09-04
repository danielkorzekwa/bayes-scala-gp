package dk.gp.cogp

import breeze.linalg.DenseMatrix
import dk.gp.cogp.svi.stochasticUpdateLB

object cogpTrain {

  def apply(x: DenseMatrix[Double], y: DenseMatrix[Double], model: CogpModel, iterNum: Int): CogpModel = {
    val finalModel = (0 until iterNum).foldLeft(model) { case (currModel, iter) => stochasticUpdateLB(currModel, x, y) }
    finalModel
  }
}