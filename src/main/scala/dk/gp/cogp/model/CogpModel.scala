package dk.gp.cogp.model

import dk.gp.cov.CovFunc
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector

case class CogpModel(x: DenseMatrix[Double], covFunc: CovFunc,covFuncParams:DenseVector[Double],modelParams:CogpModelParams) {

  val z = x //simplifying assumption

   val kXZ = covFunc.cov(x,z,covFuncParams)
  val kZZ = kXZ //simplifying assumption

}