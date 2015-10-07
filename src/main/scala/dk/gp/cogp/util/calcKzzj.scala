package dk.gp.cogp.util

import dk.gp.cogp.model.CogpModel
import breeze.linalg.DenseMatrix

object calcKzzj {

  def apply(model: CogpModel): Array[DenseMatrix[Double]] = model.g.map(g => g.calckZZ())

}