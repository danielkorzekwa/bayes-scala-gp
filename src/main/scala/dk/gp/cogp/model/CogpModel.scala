package dk.gp.cogp.model

import dk.gp.cov.CovFunc
import breeze.linalg.DenseMatrix

case class CogpModel(x: DenseMatrix[Double], covFunc: CovFunc,modelParams:CogpModelParams) {

  val z = x //simplifying assumption

  val kXZ = covFunc.covNM(x, z)
  val kZZ = kXZ //simplifying assumption

}