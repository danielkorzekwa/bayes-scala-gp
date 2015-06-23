package dk.gp.cogp

import breeze.linalg.DenseMatrix
import dk.gp.cogp.model.CogpModel
import dk.gp.cogp.model.CogpModelParams
import dk.gp.cogp.svi.optimiseLB
import dk.gp.cov.CovFunc
import breeze.linalg.DenseVector

object cogp {

  def apply(x: DenseMatrix[Double], y: DenseMatrix[Double], covFunc: CovFunc,covFuncParams:DenseVector[Double]): CogpModel = {
    
    val lbState = optimiseLB(x, y, covFunc, covFuncParams,l = 0.1, iterNum = 10)
    
    val modelParams = CogpModelParams(lbState.u,lbState.v,lbState.beta,lbState.w)
    CogpModel(x, covFunc,covFuncParams, modelParams)
  }
}