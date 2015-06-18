package dk.gp.cogp

import breeze.linalg.DenseMatrix
import dk.gp.cov.CovFunc
import dk.gp.cogp.svi.inferModelParams
import dk.gp.cogp.model.CogpModel

object cogp {

  def apply(x: DenseMatrix[Double], y: DenseMatrix[Double], covFunc: CovFunc): CogpModel = {
    
    val modelParams = inferModelParams(x, y, covFunc, l = 0.1, iterNum = 10)
    CogpModel(x, covFunc, modelParams)
  }
}