package dk.gp.cogp

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.cogp.svi.optimiseLB
import dk.gp.cov.CovFunc
import dk.gp.cov.CovFunc

object cogp {

  def apply(x: DenseMatrix[Double], y: DenseMatrix[Double], 
      covFuncG:Array[CovFunc],covFuncGParams:Array[DenseVector[Double]],
      covFuncH:Array[CovFunc],covFuncHParams:Array[DenseVector[Double]]): CogpModel = {
    
    val model = optimiseLB(x, y, covFuncG, covFuncGParams,covFuncH,covFuncHParams,l = 0.1, iterNum = 0)
    
   model
  }
}