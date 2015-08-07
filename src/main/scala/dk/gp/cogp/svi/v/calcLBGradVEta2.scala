package dk.gp.cogp.svi.v

import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix
import breeze.linalg.inv
import dk.gp.cogp.CogpModel

object calcLBGradVEta2 {

  def apply(i: Int, model: CogpModel, x: DenseMatrix[Double], y: DenseMatrix[Double]): DenseMatrix[Double] = {

    val z_i = model.h(i).z
    val kZZ_i = model.h(i).covFunc.cov(z_i, z_i, model.h(i).covFuncParams) + 1e-10 * DenseMatrix.eye[Double](x.size)
    val kXZ_i = model.h(i).covFunc.cov(z_i, z_i, model.h(i).covFuncParams)

    val Ai = kXZ_i * inv(kZZ_i)

    val v = model.h.map(_.u)
    val beta = model.beta

    val lambda = inv(kZZ_i) + beta(i) * Ai.t * Ai

    val grad = 0.5 * inv(v(i).v) - 0.5 * lambda

    grad
  }

}