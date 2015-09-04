package dk.gp.cogp.svi.v

import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix
import breeze.linalg.inv
import dk.gp.cogp.CogpModel
import breeze.linalg.cholesky
import dk.gp.math.invchol
import dk.gp.cogp.lb.LowerBound

object calcLBGradVEta2 {

  def apply(i: Int, lowerBound:LowerBound,model: CogpModel, x: DenseMatrix[Double], y: DenseMatrix[Double]): DenseMatrix[Double] = {

    val kZZ_i = lowerBound.kZZi(i)
    val kXZ_i = lowerBound.kXZi(i)

    val kZZiInv = lowerBound.kZZiInv(i)

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