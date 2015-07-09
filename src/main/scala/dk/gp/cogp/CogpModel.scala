package dk.gp.cogp

import dk.gp.cov.CovFunc
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.math.MultivariateGaussian
import dk.gp.cov.utils.covDiag

case class CogpModel(x: DenseMatrix[Double], covFunc: CovFunc,covFuncParams:DenseVector[Double],
    u:Array[MultivariateGaussian],v:Array[MultivariateGaussian],beta: DenseVector[Double],w: DenseMatrix[Double]) {

  val z = x //simplifying assumption

   val kXZ = covFunc.cov(x,z,covFuncParams)
  val kZZ = kXZ //simplifying assumption

  val kXXDiag = covDiag(x, covFunc, covFuncParams)
}