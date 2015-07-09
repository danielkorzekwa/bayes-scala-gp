package dk.gp.cogp

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.cogp.svi.optimiseLB
import dk.gp.cov.CovFunc

object cogp {

  def apply(x: DenseMatrix[Double], y: DenseMatrix[Double], covFunc: CovFunc,covFuncParams:DenseVector[Double]): CogpModel = {
    
    val lbState = optimiseLB(x, y, covFunc, covFuncParams,l = 0.1, iterNum = 0)
    
    CogpModel(x, covFunc,covFuncParams, 
        lbState.u,lbState.v,lbState.beta,lbState.w)
  }
}