package dk.gp.cogp

import breeze.linalg.DenseMatrix
import dk.gp.cogp.model.CogpModel
import dk.gp.cogp.svi.stochasticUpdateCogpModel

object cogpTrain {

  def apply(x: DenseMatrix[Double], y: DenseMatrix[Double], model: CogpModel, iterNum: Int): CogpModel = {
    val finalModel = (0 until iterNum).foldLeft(model) { case (currModel, iter) => stochasticUpdateCogpModel(currModel, x, y) }
    finalModel
  }
}