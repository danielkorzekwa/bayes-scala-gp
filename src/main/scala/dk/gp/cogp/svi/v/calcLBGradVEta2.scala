package dk.gp.cogp.svi.v

import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix
import breeze.linalg.inv
import dk.gp.cogp.CogpModel
import breeze.linalg.cholesky
import dk.gp.math.invchol

object calcLBGradVEta2 {

  def apply(i: Int, model: CogpModel, x: DenseMatrix[Double], y: DenseMatrix[Double]): DenseMatrix[Double] = {

    val z_i = model.h(i).z
    val kZZ_i = model.h(i).covFunc.cov(z_i, z_i, model.h(i).covFuncParams) + 1e-10 * DenseMatrix.eye[Double](x.size)
    val kXZ_i = model.h(i).covFunc.cov(z_i, z_i, model.h(i).covFuncParams)

    val kZZiCholR = cholesky(kZZ_i).t
    val kZZiInv = invchol(kZZiCholR)

    val Ai = kXZ_i * kZZiInv

    val v = model.h.map(_.u)
    val beta = model.beta

    val lambda = kZZiInv + beta(i) * Ai.t * Ai

    val vCholR = cholesky(v(i).v).t
    val vInv = invchol(vCholR)

    val grad = 0.5 * vInv - 0.5 * lambda

    grad
  }

}