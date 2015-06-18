package dk.gp.cogp.svi

import breeze.linalg.DenseMatrix
import dk.gp.cov.CovFunc
import dk.gp.cogp.model.CogpModelParams

object stochasticUpdateModelParams {

  def apply(modelParams: CogpModelParams, x: DenseMatrix[Double], y: DenseMatrix[Double], covFunc: CovFunc, l: Double): CogpModelParams = {

    val z = x //simplifying assumption
    val kXZ = covFunc.covNM(x, z)
    val kZZ = kXZ //simplifying assumption

    val newU = (0 until modelParams.u.size).map { j => stochasticUpdateU(j, l, modelParams, y, kXZ, kZZ) }.toArray

    val newV = (0 until modelParams.v.size).map { i => stochasticUpdateV(i, l, modelParams, y, kXZ, kZZ) }.toArray

    modelParams.copy(u = newU, v = newV)
  }

}